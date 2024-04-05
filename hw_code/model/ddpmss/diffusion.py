import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio
import speechbrain as sb
from speechbrain.pretrained import DiffWaveVocoder
from speechbrain.lobes.models.HifiGAN import mel_spectogram


#resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050)

#mixture_path = 'diffwave-example/mixture.wav'
#mixture, _16000 = torchaudio.load(mixture_path)
#assert _16000 == 16000
#mixture = resampler(mixture)  # otherwise you get funny results

device = 'cuda'
#audio = sb.dataio.dataio.read_audio(mixture_path)
#audio = torch.FloatTensor(audio).unsqueeze(0)
#audio = mixture.to(device)

#mel = mel_spectogram(sample_rate=22050, hop_length=256, win_length=1024, n_fft=1024, n_mels=80, f_min=0, f_max=8000, power=1.0, normalized=False, norm="slaney", mel_scale="slaney", compression=True, audio=audio).to(device)
diffwave = DiffWaveVocoder.from_hparams(source="speechbrain/tts-diffwave-ljspeech", savedir="tmpdir", run_opts={"device": device})

# Running Vocoder (spectrogram-to-waveform), a fast sampling can be realized by passing user-defined variance schedules. According to the paper, high-quality audios can be generated with only 6 steps (instead of a total of 50).
#waveforms = diffwave.decode_batch(
#  mel,
#  hop_len=256, # upsample factor, should be the same as "hop_len" during the extraction of mel-spectrogram
#  fast_sampling=True, # fast sampling is highly recommended
#  fast_sampling_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5], # customized noise schedule
#)

#torchaudio.save('reconstructed.wav', waveforms.squeeze(1).cpu(), 22050)


class DiffusionModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.resampler_16k22k = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050)
    self.resampler_22k16k = torchaudio.transforms.Resample(orig_freq=22050, new_freq=16000)

  @torch.inference_mode()
  def forward(self, x):
    B, C, L = x.shape
    assert C == 1
    #print(f'initially {x.shape=}')
    x = self.resampler_16k22k(x)
    #print(f'resampled to 22kHz {x.shape=}')
    x = F.pad(x, (0, (256 - x.shape[-1] % 256) % 256))  # ceils to 256
    #print(f'after padding {x.shape=}')
    mel = mel_spectogram(sample_rate=22050, hop_length=256, win_length=1024, n_fft=1024, n_mels=80, f_min=0, f_max=8000, power=1.0, normalized=False, norm="slaney", mel_scale="slaney", compression=True, audio=x)
    #print(f'here {mel.shape=}')
    assert mel.ndim == 4
    mel = mel[:, :, :, :-1]
    #print(f'{mel[0, :10, -1]=}')
    #print(f'resampled {x.shape=}, melled {mel.shape=}')
    x = torch.stack([
      diffwave.decode_batch(mel[i], hop_len=256, fast_sampling=True, fast_sampling_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5])  # this guy apparently doesn't like batches..
      for i in range(B)
    ])
    #print(f'after diffwave {x.shape=}')
    x = self.resampler_22k16k(x)
    #print(f'resampled back {x.shape=}')
    return x
