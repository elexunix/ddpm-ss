import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio
import speechbrain as sb
from speechbrain.pretrained import DiffWaveVocoder
from speechbrain.lobes.models.HifiGAN import mel_spectogram


device = 'cuda'
diffwave = DiffWaveVocoder.from_hparams(source="speechbrain/tts-diffwave-ljspeech", savedir="tmpdir", run_opts={"device": device})


class DiffusionModel(nn.Module):

  def __init__(self):
    super().__init__()

  @torch.inference_mode()
  def forward(self, x):
    B, C, L = x.shape
    assert C == 1
    x = F.pad(x, (0, (256 - x.shape[-1] % 256) % 256))  # ceils to 256
    mel = mel_spectogram(sample_rate=22050, hop_length=256, win_length=1024, n_fft=1024, n_mels=80, f_min=0, f_max=8000, power=1.0, normalized=False, norm="slaney", mel_scale="slaney", compression=True, audio=x)
    assert mel.ndim == 4
    mel = mel[:, :, :, :-1]
    x = torch.stack([
      diffwave.decode_batch(mel[i], hop_len=256, fast_sampling=True, fast_sampling_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5])  # this guy apparently doesn't like batches..
      for i in range(B)
    ])
    return x
