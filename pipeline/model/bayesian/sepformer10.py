import torch, torch.nn as nn
import speechbrain
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
from collections import OrderedDict

from speechbrain.pretrained import SepformerSeparation as sepformer
from ..sepformer10.interfaces import SepformerSeparation10


class Sepformer10ModelPretrained(nn.Module):
  def __init__(self, Nsp=10):
    super().__init__()
    self.resampler_16k8k = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000)
    self.resampler_8k16k = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)
    #sepformer3_pretrained = speechbrain.pretrained.SepformerSeparation.from_hparams(source='speechbrain/sepformer-wsj03mix', savedir='pretrained-models/sepformer-wsj03mix')
    current_dir = Path(__file__).resolve().parent
    hparams = load_hyperpyyaml(open(current_dir / 'hyperparams10.yaml'))
    #pretrained = hparams['pretrainer']
    #speechbrain.pretrained.SepformerSeparation(hparams['modules'], hparams)
    stuff = torch.load(current_dir / 'sepformer10-model563-best.pth')['state_dict']
    #print(stuff.keys())
    sepformer10_pretrained_state_dict = OrderedDict({key[6:] : value for key, value in stuff.items() if key.startswith('model.')})
    #self.model.load_state_dict(dct)
    self.model = SepformerSeparation10(sepformer10_pretrained_state_dict, modules=hparams['modules'], hparams=hparams)
    print('model with', sum(p.numel() for p in self.model.parameters()), 'parameters')
    print('training:', self.model.training, '!!')

  def forward(self, mixture_at_16khz):
    #print(f'{mixture_at_16khz.shape=}')
    mixture_at_8khz = self.resampler_16k8k(mixture_at_16khz)
    #print(f'{mixture_at_8khz.shape=}')
    est_sources, features = self.model(mixture_at_8khz)
    #print(f'{est_sources.shape=}')
    B, L, Nsp = est_sources.shape  # Sepformer-style shape
    assert Nsp == 10  # here are the 10 speakers
    predicted = est_sources.permute(0, 2, 1).contiguous()
    L2 = predicted.shape[2]
    #print(f'{predicted.shape=}, {B=}, {Nsp=}, {L2=}')
    assert predicted.shape == (B, Nsp, L2)
    return predicted, features
