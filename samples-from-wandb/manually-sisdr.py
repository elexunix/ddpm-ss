import sys
import torch, torchaudio
#from torchmetrics.functional import signal_distortion_ratio as sisdr
from torchmetrics.functional import scale_invariant_signal_distortion_ratio as sisdr
from torchaudio.transforms import Resample

assert len(sys.argv) == 3, "Usage: python3 manually-sisdr.py predicted.wav target.wav"
pred, sr_pred = torchaudio.load(sys.argv[1])
target, sr_target = torchaudio.load(sys.argv[2])
if sr_pred != sr_target:
  print(f"Warning: {sr_pred=} != {sr_target=}. Converting to {sr_target=}.")
  resampler = Resample(orig_freq=sr_pred, new_freq=sr_target)
  pred = resampler(pred)
print('SISDR:', sisdr(pred, target))
