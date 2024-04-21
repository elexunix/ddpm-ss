import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio

from .diffusion import DiffusionModel
from .sepformer import SepformerModel


class ResNetBlock(nn.Module):

  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.proj = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    self.stack = nn.Sequential(
      nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(num_features=out_channels),
      nn.ReLU(),
      nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(num_features=out_channels),
    )

  def forward(self, x):
    x = self.proj(x)
    return x + self.stack(x)


class MixerModel(nn.Module):

#  def __init__(self, in_channels, mid_channels, out_channels):
#    super().__init__()
#    self.stack = nn.Sequential(
#      ResNetBlock(in_channels=in_channels, out_channels=mid_channels),
#      ResNetBlock(in_channels=mid_channels, out_channels=out_channels),
#    )
  def __init__(self, n_channels=[32,32,64,64,64]):
    super().__init__()
    assert len(n_channels) == 5
    self.stack = nn.Sequential(
      nn.Conv2d(in_channels=2, out_channels=n_channels[0], kernel_size=3, padding=1),
      nn.BatchNorm2d(num_features=n_channels[0]),
      nn.ReLU(),
      nn.Conv2d(in_channels=n_channels[0], out_channels=n_channels[1], kernel_size=3, padding=1),
      nn.BatchNorm2d(num_features=n_channels[1]),
      nn.ReLU(),
      nn.Conv2d(in_channels=n_channels[1], out_channels=n_channels[2], kernel_size=3, padding=1),
      nn.BatchNorm2d(num_features=n_channels[2]),
      nn.ReLU(),
      nn.Conv2d(in_channels=n_channels[2], out_channels=n_channels[3], kernel_size=3, padding=1),
      nn.BatchNorm2d(num_features=n_channels[3]),
      nn.ReLU(),
      nn.Conv2d(in_channels=n_channels[3], out_channels=n_channels[4], kernel_size=3, padding=1),
      nn.BatchNorm2d(num_features=n_channels[4]),
      nn.ReLU(),
      nn.Conv2d(in_channels=n_channels[4], out_channels=2, kernel_size=3, padding=1),
    )

  def forward(self, x, y):
    B, C1, H, W = x.shape
    B_, C2, H_, W_ = y.shape
    assert B == B_ and C1 == C2 == 1 and H == H_ and W == W_, f'MixerModel received shapes {x.shape} and {y.shape}'
    #return x
    z = torch.cat([x, y], 1)
    z = self.stack(z)
    a_coeffs, b_coeffs = z[:, 0:1, :, :], z[:, 1:2, :, :]
    return a_coeffs * x + b_coeffs * y


class SepDiffModel(nn.Module):

  def __init__(self, *args, **kwargs):
    super().__init__()
    self.separator = SepformerModel()
    self.resampler_8k16k = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)
    self.resampler_8k22k = torchaudio.transforms.Resample(orig_freq=8000, new_freq=22050)
    self.resampler_22k16k = torchaudio.transforms.Resample(orig_freq=22050, new_freq=16000)
    self.denoiser = DiffusionModel()
    self.stft = lambda input : torch.stft(input.squeeze(-2), n_fft=1024, return_complex=True).unsqueeze(-3)  # sq-unsq C=1
    self.istft = lambda input : torch.istft(input.squeeze(-3), n_fft=1024, return_complex=False).unsqueeze(-2)
    self.mixer_angle = MixerModel()
    self.mixer_abs = MixerModel()
    self.dummy = nn.Parameter(torch.randn(1))

  def fix_length(self, tensor, L_target, lenience=10):
    B, C, L_old = tensor.shape
    # convs-deconvs may give, say, length 239864 instead of 239862, but we check it is not too much off:
    assert L_target - lenience <= L_old <= L_target + lenience, f'length before mixing is {L_old} but we need {L_target}, too much off'
    tensor = tensor[..., :L_target]
    tensor = F.pad(tensor, (0, L_target - tensor.shape[-1]))
    assert tensor.shape[-1] == L_target
    return tensor

  def forward(self, mixture):
    B, C, L = mixture.shape
    assert C == 1
    with torch.no_grad():
      separated1, separated2 = self.separator(mixture)
      s1, s2 = separated1.clone(), separated2.clone()
      denoised1 = self.denoiser(self.resampler_8k22k(separated1))
      denoised2 = self.denoiser(self.resampler_8k22k(separated2))
    #return {"separated1": s1, "separated2": s2, "predicted1": separated1 + 0 * self.dummy, "predicted2": separated2}  # works!
    #return {"separated1": s1, "separated2": s2, "predicted1": denoised1 + 0 * self.dummy, "predicted2": denoised2}  # doesn't work!

    with torch.no_grad():
      separated1 = self.stft(self.resampler_8k16k(separated1))
      separated2 = self.stft(self.resampler_8k16k(separated2))
      denoised1 = self.stft(self.resampler_22k16k(denoised1))
      denoised2 = self.stft(self.resampler_22k16k(denoised2))
    predicted1_angle = self.mixer_angle(separated1.angle(), denoised1.angle())
    predicted1_abs = self.mixer_abs(separated1.abs(), denoised1.abs())
    predicted2_angle = self.mixer_angle(separated2.angle(), denoised2.angle())
    predicted2_abs = self.mixer_abs(separated2.abs(), denoised2.abs())
    predicted1 = torch.view_as_complex(predicted1_abs.unsqueeze(-1) * torch.stack([torch.cos(predicted1_angle), torch.sin(predicted1_angle)], -1))
    predicted2 = torch.view_as_complex(predicted2_abs.unsqueeze(-1) * torch.stack([torch.cos(predicted2_angle), torch.sin(predicted2_angle)], -1))
    predicted1 = self.istft(predicted1)
    predicted2 = self.istft(predicted2)
    predicted1 = self.fix_length(predicted1, L, lenience=300)
    predicted2 = self.fix_length(predicted2, L, lenience=300)
    return {"separated1": s1, "separated2": s2, "predicted1": predicted1, "predicted2": predicted2}
