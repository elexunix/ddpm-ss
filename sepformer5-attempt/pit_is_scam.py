import torch, torchaudio
from torchmetrics.audio.pit import PermutationInvariantTraining as PIT
from torchmetrics.audio.sdr import scale_invariant_signal_distortion_ratio as sisdr

source1, _ = torchaudio.load('source1.wav')
source2, _ = torchaudio.load('source2.wav')
source3, _ = torchaudio.load('source3.wav')
source4, _ = torchaudio.load('source4.wav')
source5, _ = torchaudio.load('source5.wav')
sources = torch.stack([source1, source2, source3, source4, source5])
print(f'{sources.shape=}')

pit = PIT(sisdr)

print(f'{pit(sources, sources[[0, 1, 2, 3, 4]])=}')
print(f'{pit(sources, sources[[0, 1, 2, 4, 3]])=}')
print(f'{pit(sources, sources[[2, 3, 4, 1, 0]])=}')
print(f'{pit(sources, sources[[2, 3, 4, 0, 1]])=}')
print(f'{pit(sources, sources[[2, 4, 3, 0, 1]])=}')

print('IT\'S A SCAM!!')
