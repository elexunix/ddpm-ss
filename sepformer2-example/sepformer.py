import torch
import torchaudio
from torchmetrics import ScaleInvariantSignalDistortionRatio as SISDR
from torchmetrics.audio import PerceptualEvaluationSpeechQuality as PESQ, ShortTimeObjectiveIntelligibility as STOI
from speechbrain.pretrained import SepformerSeparation as sepformer


resampler_16k8k = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000)
resampler_8k16k = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)

model = sepformer.from_hparams(source='speechbrain/sepformer-wsj02mix', savedir='pretrained-models/sepformer-wsj02mix')
#model(1235)

mixture_path = 'mixture.wav'
mixture_at16khz, _16000 = torchaudio.load(mixture_path)
assert _16000 == 16000
mixture_at8khz = resampler_16k8k(mixture_at16khz)  # otherwise you get funny results
torchaudio.save('mixture-8khz.wav', mixture_at8khz, 8000)
est_sources = model(mixture_at8khz).detach().cpu()  # .separate_file
print(mixture_at8khz.shape, 'to', est_sources.shape)
predicted1_at8khz, predicted2_at8khz = est_sources[:, :, 0], est_sources[:, :, 1]
torchaudio.save('source1.wav', predicted1_at8khz, 8000)
torchaudio.save('source2.wav', predicted2_at8khz, 8000)
sisdr = SISDR()
print(f'{sisdr(predicted1_at8khz, mixture_at8khz)=}')
print(f'{sisdr(predicted2_at8khz, mixture_at8khz)=}')
pesq = PESQ(fs=16000, mode='wb', n_processes=8)
predicted1_at16khz = resampler_8k16k(predicted1_at8khz)
predicted2_at16khz = resampler_8k16k(predicted2_at8khz)
print(f'{pesq(predicted1_at16khz, mixture_at16khz)=}')
print(f'{pesq(predicted2_at16khz, mixture_at16khz)=}')
stoi = STOI(fs=16000)
print(f'{stoi(predicted1_at16khz, mixture_at16khz)=}')
print(f'{stoi(predicted2_at16khz, mixture_at16khz)=}')

from time import time
t0 = time()
sisdr(torch.tile(predicted1_at16khz, (8, 1)), torch.tile(mixture_at16khz, (8, 1)))
print('time for sisdr (bs=8): ', time() - t0)
pesq(torch.tile(predicted1_at16khz, (8, 1)), torch.tile(mixture_at16khz, (8, 1)))
print('time for pesq (bs=8): ', time() - t0)


from more_metrics import hearing_perception_metrics

#but it takes 3s for each evaluation....
print(f'{hearing_perception_metrics(predicted1_at16khz, mixture_at16khz, 16000)=}')
print(f'{hearing_perception_metrics(predicted2_at16khz, mixture_at16khz, 16000)=}')
