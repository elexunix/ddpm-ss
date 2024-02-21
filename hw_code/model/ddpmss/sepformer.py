import torch.nn as nn
import torchaudio
from speechbrain.pretrained import SepformerSeparation as sepformer


class SepformerModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.dummy = nn.Linear(0, 0)
    self.resampler_16k8k = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000)
    self.separator = sepformer.from_hparams(source='speechbrain/sepformer-wsj02mix', savedir='pretrained-models/sepformer-wsj02mix', run_opts={'device':'cuda'})
    self.resampler_8k16k = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)

  def forward(self, mixture):
    self.device = self.dummy.weight.device
    # mixture.shape: (C, 2L)
    mixture = self.resampler_16k8k(mixture).to(self.device)  # warning: otherwise you'll get funny results
    # mixture.shape: (C, L)
    source1, source2 = self.separator(mixture)
    source1, source2 = source1.T, source2.T  # now (C, L)
    source1, source2 = self.resampler_8k16k(source1), self.resampler_8k16k(source2)
    return source1, source2
