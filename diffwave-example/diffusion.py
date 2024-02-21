import torch
import torchaudio
import speechbrain as sb
from speechbrain.pretrained import DiffWaveVocoder
from speechbrain.lobes.models.HifiGAN import mel_spectogram

resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050)

mixture_path = 'mixture.wav'
mixture, _16000 = torchaudio.load(mixture_path)
assert _16000 == 16000
mixture = resampler(mixture)  # otherwise you get funny results

device = 'cuda'
#audio = sb.dataio.dataio.read_audio(mixture_path)
#audio = torch.FloatTensor(audio)
#audio = audio.unsqueeze(0)
audio = mixture.to(device)

print(f'{audio.shape=}')
audio = torch.cat([audio, audio])  # THIS LINE'S ADDED JUST TO SHOW THE CHANNELS=1 REQ.
print(f'{audio.shape=}')

mel = mel_spectogram(
  sample_rate=22050,
  hop_length=256,
  win_length=1024,
  n_fft=1024,
  n_mels=80,
  f_min=0,
  f_max=8000,
  power=1.0,
  normalized=False,
  norm="slaney",
  mel_scale="slaney",
  compression=True,
  audio=audio,
).to(device)

print(f'{mel.shape=}')

diffwave = DiffWaveVocoder.from_hparams(source="speechbrain/tts-diffwave-ljspeech", savedir="tmpdir", run_opts={"device": device})

# Running Vocoder (spectrogram-to-waveform), a fast sampling can be realized by passing user-defined variance schedules. According to the paper, high-quality audios can be generated with only 6 steps (instead of a total of 50).
waveforms = diffwave.decode_batch(
  mel,
  hop_len=256, # upsample factor, should be the same as "hop_len" during the extraction of mel-spectrogram
  fast_sampling=True, # fast sampling is highly recommended
  fast_sampling_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5], # customized noise schedule 
)

print(f'{waveforms.shape=}')

torchaudio.save('reconstructed.wav', waveforms.squeeze(1).cpu(), 22050)
