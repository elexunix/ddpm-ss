from speechbrain.pretrained import SepformerSeparation as sepformer
import torchaudio

resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000)

model = sepformer.from_hparams(source='speechbrain/sepformer-wsj02mix', savedir='pretrained-models/sepformer-wsj02mix')

mixture_path = 'mixture.wav'
mixture, _16000 = torchaudio.load(mixture_path)
assert _16000 == 16000
mixture = resampler(mixture)  # otherwise you get funny results
est_sources = model(mixture).detach().cpu()  # .separate_file
torchaudio.save('source1.wav', est_sources[:, :, 0], 8000)
torchaudio.save('source2.wav', est_sources[:, :, 1], 8000)
