import numpy as np
import matplotlib.pyplot as plt
import torch, torchaudio
from torchmetrics.audio.sdr import scale_invariant_signal_distortion_ratio as sisdr


#a = np.array([[-10.1570, -27.18069, -26.88037, -22.46214, -26.27515,
#  -22.54983, -19.43530, 0.1911354, -19.0881, -30.6235, ],
# [-25.93908, -29.05893, 1.620593, -21.01786, -26.5382,
#  -18.92169, -19.13370, -25.19915, -16.10532, -27.70341, ],
# [ -8.70029, -14.74300, -13.65674, -9.176805, -18.60657,
#  -11.0664, -8.18568, -12.116261, -10.36758, -14.02402, ],
# [-18.71333, -16.25730, -23.50342, -20.06373, -29.43488,
#  -20.28203, -19.36111, -20.48458, -23.39163, 0.57990426],
# [-21.58161, -29.75676, -25.34602, -20.60702, 3.782342,
#  -20.46479, -28.57196, -27.6773, -26.18349, -29.65745, ],
# [-19.08666, -1.593911, -28.7600, -18.49780, -28.59006,
#  -23.28140, -18.47295, -21.30115, -20.23244, -19.31249, ],
# [-24.19918, -24.95380, -22.53117, -11.01284, -27.42562,
#   -3.474664, -21.54321, -21.84110, -23.98847, -22.46852, ],
# [ -9.83906, -21.06691, -23.68426, -22.04088, -31.83748,
#  -23.7802, -14.12762, -17.25427, -1.615886, -26.11979, ],
# [ -4.991492, -19.69994, -13.342916, -18.12483, -15.931643,
#  -10.4349, -10.32419, -10.6520, -11.96357, -19.96168, ],
# [ -6.78232, -16.13167, -15.87575, -14.66383, -17.44607,
#  -10.87928, -10.02706, -9.84341, -14.04715, -14.806313, ],])
#
#print(a)
#assert a.shape == (10, 10)
#plt.imshow(a)
#plt.savefig('1/sisdrs10.png')


#preds, tgts = [], []
#for i in range(10):
#  pred, sr = torchaudio.load(f'2/predicted{i}.wav')
#  assert sr == 16000
#  preds.append(pred)
#for i in range(10):
#  tgt, sr = torchaudio.load(f'2/target{i}.wav')
#  assert sr == 16000
#  tgts.append(tgt)
#
#a = np.array([[sisdr(pred, tgt).item() for tgt in tgts] for pred in preds])
#
#print(a)
#assert a.shape == (10, 10)
#plt.imshow(a)
#plt.savefig('2/sisdrs10.png')


preds, tgts = [], []
for i in range(10):
  pred, sr = torchaudio.load(f'2/predicted{i}.wav')
  assert sr == 16000
  pred *= 20 / pred.norm()
  torchaudio.save(f'2/normalized/predicted{i}.wav', pred, sr)
  preds.append(pred)
for i in range(10):
  tgt, sr = torchaudio.load(f'2/target{i}.wav')
  assert sr == 16000
  tgt *= 20 / tgt.norm()
  torchaudio.save(f'2/normalized/target{i}.wav', tgt, sr)
  tgts.append(tgt)

a = np.array([[sisdr(pred, tgt).item() for tgt in tgts] for pred in preds])

print(a)
assert a.shape == (10, 10)
plt.imshow(a)
plt.savefig('2/normalized/sisdrs10.png')
