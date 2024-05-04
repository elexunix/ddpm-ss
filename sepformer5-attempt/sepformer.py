import torch, torch.nn as nn
import torchaudio
#import speechbrain.pretrained
#from speechbrain.pretrained import SepformerSeparation as sepformer
#import speechbrain
from hyperpyyaml import load_hyperpyyaml
from collections import OrderedDict
from pathlib import Path

from pipeline.model.sepformer5.interfaces import SepformerSeparation5


#model = speechbrain.pretrained.SepformerSeparation.from_hparams(source='speechbrain/sepformer-wsj03mix', savedir='pretrained-models/sepformer-wsj03mix')

#hparams = load_hyperpyyaml(open('hyperparams.yaml'))
#pretrained = hparams['pretrainer']
#speechbrain.pretrained.SepformerSeparation(hparams['modules'], hparams)


#class Pretrained(nn.Module):
#  def __init__(self, modules, hparams):
#    super().__init__()
#    self.mods = nn.ModuleDict(modules)
#    self.device = 'cpu'
#    for module in self.mods.values():
#      if module is not None:
#        module.to(self.device)
#
#
#class SepformerModel(Pretrained):
#  def __init__(self, hparams):
#    super().__init__(modules=hparams['modules'], hparams=hparams)
#    stuff = torch.load('sepformer5-model329-best.pth')['state_dict']
#    #print(stuff.keys())
#    dct = OrderedDict({key[6:] : value for key, value in stuff.items() if key.startswith('model.')})
#    self.load_state_dict(dct)
#  def separate_batch(self, mix):
#    raise NotImplementedError("speechbrain/pretrained/interfaces.py:2184")
#  def forward(self, mix):
#    return self.separate_batch(mix)
#
#
#model = SepformerModel(hparams)
#print('model with', sum(p.numel() for p in model.parameters()), 'parameters')
#print('training:', model.training, '!!')


class Sepformer5ModelPretrained(nn.Module):
  def __init__(self):
    super().__init__()
    self.resampler_16k8k = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000)
    self.resampler_8k16k = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)
    #sepformer3_pretrained = speechbrain.pretrained.SepformerSeparation.from_hparams(source='speechbrain/sepformer-wsj03mix', savedir='pretrained-models/sepformer-wsj03mix')
    current_dir = Path(__file__).resolve().parent
    hparams = load_hyperpyyaml(open(current_dir / 'hyperparams.yaml'))
    #pretrained = hparams['pretrainer']
    #speechbrain.pretrained.SepformerSeparation(hparams['modules'], hparams)
    stuff = torch.load('sepformer5-model329-best.pth')['state_dict']
    print(stuff.keys())
    sepformer5_pretrained_state_dict = OrderedDict({key[6:] : value for key, value in stuff.items() if key.startswith('model.')})
    #self.model.load_state_dict(dct)
    self.model = SepformerSeparation5(sepformer5_pretrained_state_dict, modules=hparams['modules'], hparams=hparams)
    print('model with', sum(p.numel() for p in self.model.parameters()), 'parameters')
    print('training:', self.model.training, '!!')

  def forward(self, mixture_at_16khz):
    print('DEVICE', mixture_at_16khz.device)
    print(f'{mixture_at_16khz.shape=}')
    unsqueezed = mixture_at_16khz.ndim == 2
    #if unsqueezed:
    mixture_at_16khz = mixture_at_16khz.unsqueeze(-2)
    mixture_at_8khz = self.resampler_16k8k(mixture_at_16khz)
    print(f'{mixture_at_8khz.shape=}')
    est_sources = self.model(mixture_at_8khz)
    print(f'{est_sources.shape=}')
    B, L, Nsp = est_sources.shape  # Sepformer-style shape
    assert Nsp == 5  # here are the 5 speakers
    predicted = self.resampler_8k16k(est_sources.permute(-1, 0, 1).contiguous()).permute(1, 2, 0)
    L2 = predicted.shape[1]
    #assert predicted.shape == (Nsp, B, 1, L2)
    print(f'{predicted.shape=}, {B=}, {L2=}, {Nsp=}')
    assert predicted.shape == (B, L2, Nsp)
    #if unsqueezed:
    #  predicted = predicted.squeeze(-2)
    return predicted
