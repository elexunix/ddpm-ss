import torch, torch.nn as nn
import torchaudio
from speechbrain.pretrained import SepformerSeparation as sepformer


device = 'cuda'

separator = sepformer.from_hparams(source='speechbrain/sepformer-wsj02mix', savedir='pretrained-models/sepformer-wsj02mix', run_opts={'device':device})


class SepformerModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.dummy = nn.Linear(0, 0)
    self.resampler_16k8k = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000)

  @torch.inference_mode()
  def forward(self, mixture):
    self.device = self.dummy.weight.device
    B, C, L2 = mixture.shape  # (B, C=1, 2*L)
    assert C == 1
    mixture = self.resampler_16k8k(mixture).to(self.device)  # warning: otherwise you'll get funny results
    # mixture.shape: (B, C=1, L)
    sources = separator(mixture[..., 0, :])  # separator : (B, L) -> (B, L, 2)  \o_O/
    source1 = sources[..., None, :, 0]  # (B, C=1, L)
    source2 = sources[..., None, :, 1]   # (B, C=1, L)
    return source1, source2
