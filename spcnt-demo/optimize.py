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

for lr in tqdm(10**np.linspace(0, -3, 1000)):
  n_sp_real = model(audio)
  if np.random.rand() < .1:
    print(n_sp_real.item())
  direction_towards = torch.autograd.grad((n_sp_real - 1)**2, audio)[0]
  audio.requires_grad_(False)
  audio -= lr * direction_towards
  audio.requires_grad_(True)
torchaudio.save('kek-opinion-646.wav', audio[0], 16000)
