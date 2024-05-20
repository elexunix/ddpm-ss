import numpy as np
import torch, torchaudio
import matplotlib.pyplot as plt

d, sr = torchaudio.load('p.wav')
assert sr == 16000
s, sr = torchaudio.load('s.wav')
assert sr == 16000
t, sr = torchaudio.load('t.wav')
assert sr == 16000

s = s[0] / s.norm()
d = d[0] / d.norm()
t = t[0] / t.norm()

plt.plot(s, label='after sepformer')
plt.plot(d, label='after diffwave')
plt.plot(t, label='target')
plt.legend()
plt.show()
