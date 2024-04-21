#from speechbrain.pretrained import SepformerSeparation as sepformer
from .interfaces import SepformerSeparation5
import torchaudio


class Sepformer5Model():
  def __init__(self):
    self.resampler_16k8k = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000)
    self.resampler_8k16k = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)
    self.model = SepformerSeparation5.from_hparams(source='speechbrain/sepformer-wsj03mix', savedir='pretrained-models/sepformer-wsj03mix')

  def forward(self, mixture_at_16khz):
    mixture_at_8khz = self.resampler16k8k(mixture_at_16khz)
    est_sources = self.model(mixture_at_8khz)
    assert est_sources.ndim == 3
    assert est_sources.shape[-1] == 5  # here are the 5 speakers
    predicted = self.resampler_8k16k(est_sources.permute(-1, 0, 1))
    return {f'predicted{i}' : predicted[i] for i in range(5)}
