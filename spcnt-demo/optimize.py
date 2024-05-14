import numpy as np, torch, torchaudio
from pipeline.model.spcnt.model import SpeakerCountingModel as Model
from tqdm import tqdm, trange

stuff = torch.load('spcnt-run646-best.pth')['state_dict']
model = Model()
model.load_state_dict(stuff)

audio, sr = torchaudio.load('epic-0sp-pred1.28.wav')
assert sr == 16000
audio = audio[None]
audio.requires_grad_()
print(f'{audio.shape=}')

# 1e2 perfectly ok, 1e3 ended in 1.2, 1e4 exploded
for lr in tqdm([0.1] * 5):
  n_sp_real = model(audio)
  print(n_sp_real.item())
  direction_towards = torch.autograd.grad((n_sp_real - 3)**2, audio)[0]
  audio.requires_grad_(False)
  audio -= lr * direction_towards
  audio.requires_grad_(True)
torchaudio.save('kek-opinion-646.wav', audio[0], 16000)
