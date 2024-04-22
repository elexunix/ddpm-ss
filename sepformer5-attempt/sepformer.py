import torch, torch.nn as nn
import speechbrain.pretrained
from hyperpyyaml import load_hyperpyyaml


#model = speechbrain.pretrained.SepformerSeparation.from_hparams(source='speechbrain/sepformer-wsj03mix', savedir='pretrained-models/sepformer-wsj03mix')

hparams = load_hyperpyyaml(open('hyperparams.yaml'))
#pretrained = hparams['pretrainer']
#speechbrain.pretrained.SepformerSeparation(hparams['modules'], hparams)


class Pretrained(nn.Module):
  def __init__(self, modules, hparams):
    super().__init__()
    self.mods = nn.ModuleDict(modules)
    self.device = 'cpu'
    for module in self.mods.values():
      if module is not None:
        module.to(self.device)


class SepformerModel(Pretrained):
  def __init__(self, hparams):
    super().__init__(modules=hparams['modules'], hparams=hparams)
  def separate_batch(self, mix):
    raise NotImplementedError("speechbrain/pretrained/interfaces.py:2184")
  def forward(self, mix):
    return self.separate_batch(mix)


model = SepformerModel(hparams)
print('model with', sum(p.numel() for p in model.parameters()), 'parameters')
print('training:', model.training, '!!')
