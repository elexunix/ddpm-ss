from speechbrain.pretrained import SepformerSeparation as sepformer
import torchaudio
from torchmetrics import ScaleInvariantSignalDistortionRatio as SISDR

resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000)

model = sepformer.from_hparams(source='speechbrain/sepformer-wsj03mix', savedir='pretrained-models/sepformer-wsj03mix')

mixture_path = 'mixture.wav'
mixture, _16000 = torchaudio.load(mixture_path)
assert _16000 == 16000
mixture = resampler(mixture)  # otherwise you get funny results
est_sources = model(mixture).detach().cpu()  # .separate_file
print(mixture.shape, 'to', est_sources.shape)
predicted1, predicted2, predicted3 = est_sources[:, :, 0], est_sources[:, :, 1], est_sources[:, :, 2]
torchaudio.save('source1.wav', predicted1, 8000)
torchaudio.save('source2.wav', predicted2, 8000)
torchaudio.save('source3.wav', predicted3, 8000)
sisdr = SISDR()
print(f'{sisdr(predicted1, mixture)=}')
print(f'{sisdr(predicted2, mixture)=}')
print(f'{sisdr(predicted3, mixture)=}')
