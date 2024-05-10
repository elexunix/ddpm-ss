import torch, torch.nn as nn
from speechbrain.pretrained import SepformerSeparation as sepformer
from .interfaces import SepformerSeparation10
import speechbrain
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path


class Sepformer10Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.resampler_16k8k = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000)
    self.resampler_8k16k = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)

    sepformer3_pretrained = speechbrain.pretrained.SepformerSeparation.from_hparams(source='speechbrain/sepformer-wsj03mix', savedir='pretrained-models/sepformer-wsj03mix')
    current_dir = Path(__file__).resolve().parent
    hparams = load_hyperpyyaml(open(current_dir / 'hyperparams.yaml'))
    #pretrained = hparams['pretrainer']
    #speechbrain.pretrained.SepformerSeparation(hparams['modules'], hparams)
    self.model = SepformerSeparation10(sepformer3_pretrained.state_dict(), modules=hparams['modules'], hparams=hparams)
    print('model with', sum(p.numel() for p in self.model.parameters()), 'parameters')
    print('training:', self.model.training, '!!')

  def forward(self, mixture_at_16khz):
    mixture_at_8khz = self.resampler_16k8k(mixture_at_16khz)
    est_sources = self.model(mixture_at_8khz)
    B, L, Nsp = est_sources.shape  # Sepformer-style shape
    assert Nsp == 10  # here are the 10 speakers
    predicted = self.resampler_8k16k(est_sources.permute(-1, 0, 1).contiguous()).unsqueeze(-2)
    L2 = predicted.shape[-1]
    assert predicted.shape == (Nsp, B, 1, L2)
    return {f'predicted{i+1}' : predicted[i] for i in range(10)}
