from pathlib import Path
from random import shuffle
import PIL
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio
from torchvision.transforms import ToTensor
#from torchmetrics import ScaleInvariantSignalDistortionRatio as SISDR  # it's a scam; for more scam, please visit https://github.com/elexunix/scam
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality as PESQ
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility as STOI
from torchmetrics.functional import scale_invariant_signal_distortion_ratio as SISDR
from tqdm import tqdm
import itertools, multiprocessing

from pipeline.base import BaseTrainer
from pipeline.base.base_text_encoder import BaseTextEncoder
from pipeline.logger.utils import plot_spectrogram_to_buf
from pipeline.metric.utils import calc_cer, calc_wer
from pipeline.utils import inf_loop, MetricTracker


class DDPMSSTrainer(BaseTrainer):
  '''
  DDPM-based Source Separation Trainer class
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
    self.train_metrics_sometimes = MetricTracker('pesq', 'stoi', *[m.name for m in self.metrics], writer=self.writer)  # not used
    self.evaluation_metrics = MetricTracker('sisdr', 'pesq', 'stoi', 'loss', *[m.name for m in self.metrics], writer=self.writer)
    #self.sisdr = SISDR().to(device)
    self.sisdr = SISDR
    self.all_permutations = torch.tensor(list(itertools.permutations(range(self.Nsp))), device=self.device)
    print(f'{self.all_permutations.shape=}')
    self.pesq = PESQ(fs=16000, mode='wb')
    self.stoi = STOI(fs=16000)

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
      compute_aux=not is_train
    ))
    if is_train:
      batch['loss'].backward()
      self._clip_grad_norm()
      self.optimizer.step()
      if self.lr_scheduler is not None:
        self.lr_scheduler.step()

    metrics.update('loss', batch['loss'].item())
    metrics.update('sisdr', batch['sisdr'].item())
    if not is_train:
      metrics.update('pesq', batch['pesq'].item())
      metrics.update('stoi', batch['stoi'].item())
    #metrics.update('clf accuracy', batch['clf_accuracy'].item())
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

#  def _log_predictions(
#      self,
#      text,
#      log_probs,
#      log_probs_length,
#      audio_path,
#      examples_to_log=10,
#      *args,
#      **kwargs,
#  ):
#    if self.writer is None:
#      return
#    argmax_inds = log_probs.cpu().argmax(-1).numpy()
#    argmax_inds = [
#      inds[: int(ind_len)]
#      for inds, ind_len in zip(argmax_inds, log_probs_length.numpy())
#    ]
#    argmax_texts_raw = [self.text_encoder.decode(inds) for inds in argmax_inds]
#    argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]
#    tuples = list(zip(argmax_texts, text, argmax_texts_raw, audio_path))
#    random.shuffle(tuples)
#    rows = {}
#    for pred, target, raw_pred, audio_path in tuples[:examples_to_log]:
#      target = BaseTextEncoder.normalize_text(target)
#      wer = calc_wer(target, pred) * 100
#      cer = calc_cer(target, pred) * 100
#
#      rows[Path(audio_path).name] = {
#        'target': target,
#        'raw prediction': raw_pred,
#        'predictions': pred,
#        'wer': wer,
#        'cer': cer,
#      }
#    self.writer.add_table('predictions', pd.DataFrame.from_dict(rows, orient='index'))

  def _log_spectrogram(self, caption, spectrogram_batch):
    return
    spectrogram = spectrogram[0].cpu()  #random.choice(spectrogram_batch.cpu())
    image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
    self.writer.add_image(caption, ToTensor()(image))

  def _log_audio(self, caption, audio_batch):
    audio = audio_batch.cpu()[0]  #random.choice(audio_batch.cpu())
    audio *= 30 / audio.norm(dim=-1, keepdim=True)
    self.writer.add_audio(caption, audio, sample_rate=16000)

  def _log_audios(self, batch):
    self._log_audio('mixed spectrogram', batch['mixed'])
    for i in range(self.Nsp):
      self._log_audio(f'target{i+1} spectrogram', batch[f'target{i+1}'])
      self._log_audio(f'predicted{i+1} spectrogram', batch[f'target{i+1}'])
      self._log_audio(f'separated{i+1}_prebridge spectrogram', batch[f'separated{i+1}_prebridge'])
      self._log_audio(f'separated{i+1}_postbridge spectrogram', batch[f'separated{i+1}_postbridge'])
      self._log_audio(f'denoised{i+1}_prebridge spectrogram', batch[f'denoised{i+1}_prebridge'])
      self._log_audio(f'denoised{i+1}_postbridge spectrogram', batch[f'denoised{i+1}_postbridge'])

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

  def compute_metrics(self, preds, tgts, compute_aux):
    assert type(preds) is torch.Tensor and type(tgts) is torch.Tensor
    _, B, _, L = preds.shape
    assert preds.shape == tgts.shape == (self.Nsp, B, 1, L)
    def metric_matrix(metric_fn):
      return torch.stack([
        #metric_fn(pred[i:i+1].repeat(self.Nsp, 1, 1, 1), tgt)  # pesq reduce is hard-coded, rendering it unusable in my case
        torch.stack([metric_fn(pred, tgt).squeeze(-1) for tgt in tgts]) for pred in preds
      ])
    def metric_matrix_parallel_nograd(metric_fn):
      with torch.no_grad():
        with multiprocessing.Pool(24) as p:
          P = preds[None, :, ...].repeat(self.Nsp, 1, 1, 1, 1).view(self.Nsp**2 * B, *preds.shape[-2:]).cpu()
          T = tgts[:, None, ...].repeat(1, self.Nsp, 1, 1, 1).view(self.Nsp**2 * B, *tgts.shape[-2:]).cpu()
          assert P.shape == T.shape == (self.Nsp**2 * B, 1, L)
          M = torch.stack(p.starmap(metric_fn, zip(P, T)))
          return M.view(self.Nsp, self.Nsp, B)
    sisdr_matrix = metric_matrix(self.sisdr)
    #print(f'{sisdr_matrix.shape=}')
    if compute_aux:
      pesq_matrix = metric_matrix_parallel_nograd(self.pesq)
      stoi_matrix = metric_matrix_parallel_nograd(self.stoi)
      #print(f'{pesq_matrix.shape=}, {stoi_matrix.shape=}')
    if np.random.rand() < 1e-3:
      print(sisdr_matrix[:, :, 0].cpu().detach().numpy())
    assert sisdr_matrix.shape == (self.Nsp, self.Nsp, B)
#    assert not compute_aux or pesq_matrix.shape == stoi_matrix.shape == (self.Nsp, self.Nsp, B)
#    sisdr, sisdr_argmax = torch.stack([
#      sum(sisdr_matrix[i, sigma[i]] for i in range(self.Nsp)) / self.Nsp
#      for sigma in itertools.permutations(range(self.Nsp))
#    ]).max(0)  # max_{over matchings} average_{in permutation} SISDR
#    all_permutations = torch.tensor(list(itertools.permutations(range(self.Nsp))))
#    sigmas = all_permutations[sisdr_argmax.cpu()]
#    if compute_aux:
#      pesq = torch.stack([sum(pesq_matrix[spi, sigma[spi], s] for spi in range(self.Nsp)) / self.Nsp for s, sigma in enumerate(sigmas)])
#      stoi = torch.stack([sum(stoi_matrix[spi, sigma[spi], s] for spi in range(self.Nsp)) / self.Nsp for s, sigma in enumerate(sigmas)])
#    assert sisdr.shape == (B,)
#    assert not compute_aux or pesq.shape == stoi.shape == (B,)
    sisdr, sisdr_argmax = (torch.einsum('aiib', sisdr_matrix[self.all_permutations]) / self.Nsp).max(0)
    assert sisdr.shape == (B,)
    sigmas = self.all_permutations[sisdr_argmax].cpu()
    assert sigmas.shape == (B, self.Nsp)
    if compute_aux:
      pesq = torch.stack([pesq_matrix[torch.arange(self.Nsp), sigmas[i], i].mean(0) for i in range(B)])
      stoi = torch.stack([stoi_matrix[torch.arange(self.Nsp), sigmas[i], i].mean(0) for i in range(B)])
      #print(f'{pesq_matrix.shape=}, {stoi_matrix.shape=}, {torch.arange(self.Nsp).shape=}, {sigmas.shape=}, {pesq.shape=}, {stoi.shape=}')
      assert pesq.shape == stoi.shape == (B,)
    #print(f'final {sisdr=}')
    return {
      'sisdr': sisdr.mean(),
      'pesq': pesq.mean() if compute_aux else None,
      'stoi': stoi.mean() if compute_aux else None,
      'loss': -sisdr.mean()
    }
