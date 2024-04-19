import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio
import speechbrain as sb
#from speechbrain.pretrained import DiffWaveVocoder
from hw_code.model.bayesian.interfaces import DiffWaveConditionalInferer
from speechbrain.lobes.models.HifiGAN import mel_spectogram
from .diffwave import DiffWaveDiffusionTuned


device = 'cuda'
#diffwave = DiffWaveVocoder.from_hparams(source="speechbrain/tts-diffwave-ljspeech", savedir="tmpdir", run_opts={"device": device})


class DiffusionModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.diffwave = DiffWaveConditionalInferer.from_hparams(source="speechbrain/tts-diffwave-ljspeech", savedir="hw_code/model/bayesian/tmpdir", run_opts={"device": device})

  @torch.inference_mode()
  def forward(self, xs, m0):
    B, C, L = xs.shape  # x: predicted sources (by backbone), we condition on them
    assert C == 2
    assert m0.shape == (B, 1, L)  # m0: expected sum of sources, we condition on it
    #xs = F.pad(xs, (0, (256 - xs.shape[-1] % 256) % 256))  # ceils to 256
    mels = torch.stack([mel_spectogram(sample_rate=22050, hop_length=256, win_length=1024, n_fft=1024, n_mels=80, f_min=0, f_max=8000, power=1.0, normalized=False, norm="slaney", mel_scale="slaney", compression=True, audio=x_) for x_ in xs])
    assert mels.ndim == 4
    mels = mels[..., :-1]
    #x = torch.stack([
    #  diffwave.decode_batch(mel[i], hop_len=256, fast_sampling=True, fast_sampling_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5])  # this guy apparently doesn't like batches..
    #  for i in range(B)
    #])
    xs = self.diffwave.decode_batch(mels, m0, xs, hop_len=256, fast_sampling=True, fast_sampling_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5])  # this guy likes batches
    return xs
