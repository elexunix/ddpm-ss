from pathlib import Path
from random import shuffle
import PIL
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio
from torchvision.transforms import ToTensor
#from torchmetrics import ScaleInvariantSignalDistortionRatio as SISDR  # it's a scam; for more scam, please visit https://github.com/elexunix/scam
from torchmetrics.functional import scale_invariant_signal_distortion_ratio as SISDR
from tqdm import tqdm
import itertools

from pipeline.base import BaseTrainer
from pipeline.base.base_text_encoder import BaseTextEncoder
from pipeline.logger.utils import plot_spectrogram_to_buf
from pipeline.metric.utils import calc_cer, calc_wer
from pipeline.utils import inf_loop, MetricTracker


class SepformerNTrainer(BaseTrainer):
  '''
  Sepformer3-based SepformerN Model trainer class
  '''

  def __init__(
      self,
      Nsp,
      model,
      metrics,
      optimizer,
      config,
      device,
      dataloaders,
      lr_scheduler=None,
      len_epoch=None,
      skip_oom=True,
  ):
    super().__init__(model, None, metrics, optimizer, config, device)
    self.Nsp = Nsp
    self.skip_oom = skip_oom
    self.config = config
    self.train_dataloader = dataloaders['train']
    if len_epoch is None:
      # epoch-based training
      self.len_epoch = len(self.train_dataloader)
    else:
      # iteration-based training
      self.train_dataloader = inf_loop(self.train_dataloader)
      self.len_epoch = len_epoch
    self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != 'train'}
    self.lr_scheduler = lr_scheduler
    self.log_step = self.config['trainer']['log_interval']
    self.train_metrics = MetricTracker('sisdr', 'loss', 'grad norm', *[m.name for m in self.metrics], writer=self.writer)
    self.evaluation_metrics = MetricTracker('sisdr', 'loss', *[m.name for m in self.metrics], writer=self.writer)
    #self.sisdr = SISDR().to(device)
    self.sisdr = SISDR
    self.all_permutations = torch.tensor(list(itertools.permutations(range(self.Nsp))), device=self.device)
    print(f'{self.all_permutations.shape=}')

  @staticmethod
  def move_batch_to_device(batch, device: torch.device):
    #for tensor_for_gpu in ['mixed'] + [f'target{i+1}' for i in range(self.Nsp)]:
    for tensor_for_gpu in batch.keys():
      batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
    return batch

  def _clip_grad_norm(self):
    if self.config['trainer'].get('grad_norm_clip', None) is not None:
      nn.utils.clip_grad_norm_(
        self.model.parameters(), self.config['trainer']['grad_norm_clip']
      )

  def _train_epoch(self, epoch):
    '''
    Training logic for an epoch

    :param epoch: Integer, current training epoch.
    :return: A log that contains average loss and metric in this epoch.
    '''
    self.model.train()
    self.train_metrics.reset()
    self.writer.add_scalar('epoch', epoch)
    for batch_idx, batch in enumerate(
        tqdm(self.train_dataloader, desc='train', total=self.len_epoch)
    ):
      try:
        batch = self.process_batch(
          batch,
          is_train=True,
          metrics=self.train_metrics,
        )
      except RuntimeError as e:
        if 'out of memory' in str(e) and self.skip_oom:
          self.logger.warning('OOM on batch. Skipping batch.')
          for p in self.model.parameters():
            if p.grad is not None:
              del p.grad # free some memory
          torch.cuda.empty_cache()
          continue
        else:
          raise e
      self.train_metrics.update('grad norm', self.get_grad_norm())
      if batch_idx % self.log_step == 0:
        self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
        self.logger.debug(
          f'Train Epoch: {epoch} {self._progress(batch_idx)} Loss: {batch["loss"].item():.6f}'
        )
        self.writer.add_scalar(
          'learning rate', self.lr_scheduler.get_last_lr()[0]
        )
        #self._log_predictions(**batch)
        self._log_audios(batch)
        self._log_scalars(self.train_metrics)
        # we don't want to reset train metrics at the start of every epoch
        # because we are interested in recent train metrics
        last_train_metrics = self.train_metrics.result()
        self.train_metrics.reset()
      if batch_idx >= self.len_epoch:
        break
    log = last_train_metrics

    for part, dataloader in self.evaluation_dataloaders.items():
      val_log = self._evaluation_epoch(epoch, part, dataloader)
      log.update(**{f'{part}_{name}': value for name, value in val_log.items()})

    return log

  def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
    batch = self.move_batch_to_device(batch, self.device)
    if is_train:
      self.optimizer.zero_grad()
    outputs = self.model(batch['mixed'])
    if type(outputs) is dict:
      batch.update(outputs)
    else:
      raise Exception("kek")

    batch.update(self.compute_metrics(
      torch.stack([outputs[f'predicted{i+1}'] for i in range(self.Nsp)]),
      torch.stack([batch[f'target{i+1}'] for i in range(self.Nsp)]),
    ))
    if is_train:
      batch['loss'].backward()
      self._clip_grad_norm()
      self.optimizer.step()
      if self.lr_scheduler is not None:
        self.lr_scheduler.step()

    metrics.update('loss', batch['loss'].item())
    metrics.update('sisdr', batch['sisdr'].item())
    for met in self.metrics:
      metrics.update(met.name, met(**batch))
    return batch

  def _evaluation_epoch(self, epoch, part, dataloader):
    '''
    Validate after training an epoch

    :param epoch: Integer, current training epoch.
    :return: A log that contains information about validation
    '''
    self.model.eval()
    self.evaluation_metrics.reset()
    with torch.no_grad():
      for batch_idx, batch in tqdm(enumerate(dataloader), desc=part, total=len(dataloader)):
        batch = self.process_batch(batch, is_train=False, metrics=self.evaluation_metrics)
      self.writer.set_step(epoch * self.len_epoch, part)
      self._log_scalars(self.evaluation_metrics)
      #self._log_predictions(**batch)
      self._log_audios(batch)

    # DON'T add histogram of model parameters to the tensorboard
    #for name, p in self.model.named_parameters():
    #  self.writer.add_histogram(name, p, bins='auto')
    return self.evaluation_metrics.result()

  def _progress(self, batch_idx):
    base = '[{}/{} ({:.0f}%)]'
    if hasattr(self.train_dataloader, 'n_samples'):
      current = batch_idx * self.train_dataloader.batch_size
      total = self.train_dataloader.n_samples
    else:
      current = batch_idx
      total = self.len_epoch
    return base.format(current, total, 100.0 * current / total)

  #def _log_spectrogram(self, caption, spectrogram_batch):
  #  return
  #  spectrogram = spectrogram[0].cpu()  #random.choice(spectrogram_batch.cpu())
  #  image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
  #  self.writer.add_image(caption, ToTensor()(image))

  def _log_audio(self, caption, audio_batch):
    audio = audio_batch.cpu()[0]  #random.choice(audio_batch.cpu())
    self.writer.add_audio(caption, audio, sample_rate=16000)

  def _log_audios(self, batch):
    self._log_audio('mixed spectrogram', batch['mixed'])
    for i in range(self.Nsp):
      self._log_audio(f'target{i+1} spectrogram', batch[f'target{i+1}'])
      self._log_audio(f'predicted{i+1} spectrogram', batch[f'predicted{i+1}'])

  @torch.no_grad()
  def get_grad_norm(self, norm_type=2):
    parameters = self.model.parameters()
    if isinstance(parameters, torch.Tensor):
      parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.norm(
      torch.stack(
        [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
      ),
      norm_type,
    )
    return total_norm.item()

  def _log_scalars(self, metric_tracker: MetricTracker):
    if self.writer is None:
      return
    for metric_name in metric_tracker.keys():
      self.writer.add_scalar(f'{metric_name}', metric_tracker.avg(metric_name))

  def mask_length(self, xs, lengths):
    assert len(xs) == len(lengths)
    result = torch.zeros_like(xs)
    for i, l in enumerate(lengths):
      result[i, :l] = xs[i, :l]
    return result

  def compute_metrics(self, preds, tgts):
    assert type(preds) is torch.Tensor and type(tgts) is torch.Tensor
    _, B, _, L = preds.shape
    assert preds.shape == tgts.shape == (self.Nsp, B, 1, L)
    sisdr_matrix = torch.stack([
      torch.stack([self.sisdr(preds[i], tgts[j]) for j in range(self.Nsp)])
      for i in range(self.Nsp)
    ]).squeeze(-1).to(self.device)
    if np.random.rand() < 1e-3:
      print(sisdr_matrix[:, :, 0].cpu().detach().numpy())
    #print(f'{sisdr_matrix.shape=}')
    B = sisdr_matrix.shape[2]
    assert sisdr_matrix.shape == (self.Nsp, self.Nsp, B)
    #print('C', sisdr_matrix.shape, torch.stack([
    #  sum(sisdr_matrix[i, sigma[i]] for i in range(self.Nsp)) / self.Nsp
    #  for sigma in itertools.permutations(range(self.Nsp))
    #]).shape)
    # "C torch.Size([5, 5, 4, 1]) torch.Size([120, 4, 1])"
    #print(f'{sisdr_matrix[:, :, 0]=}')
    #print(f'{sisdr_matrix[self.all_permutations].shape=}')
    #print(f'{torch.einsum("aiib", sisdr_matrix[self.all_permutations]).shape=}')
    sisdr = torch.einsum('aiib', sisdr_matrix[self.all_permutations]).max(0)[0]
    #print(f'by samples sum {sisdr=}')
    sisdr = sisdr.mean(-1) / self.Nsp
    #print(f'final {sisdr=}')
    return {'sisdr': sisdr, 'loss': -sisdr}
