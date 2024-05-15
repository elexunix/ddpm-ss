import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio

from .diffusion import DiffusionModel
#from .sepformer import SepformerModel
#from .sepformer5 import Sepformer5ModelPretrained as SepformerModel
from .sepformer10 import Sepformer10ModelPretrained as SepformerModel


class MixerModel(nn.Module):

  def __init__(self, Nsp, n_channels=[32,32,64,64,64]):
    super().__init__()
    self.Nsp = Nsp
    assert len(n_channels) == 5
    self.mixer_stack = nn.Sequential(
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
    B, C, H, W = x.shape
    B_, C_, H_, W_ = y.shape
    assert B == B_ and C == C_ == self.Nsp and H == H_ and W == W_, f'MixerModel received shapes {x.shape} and {y.shape}'
    #return x
    result_slices = []
    for i in range(self.Nsp):
      x_slice, y_slice = x[:, i, :, :], y[:, i, :, :]
      z = torch.stack([x_slice, y_slice], 1)
      z = self.mixer_stack(z)
      assert z.ndim == 4
      assert z.shape[1] == 2
      a_coeffs, b_coeffs = z[:, 0, :, :], z[:, 1, :, :]
      result_slices.append(a_coeffs * x_slice + b_coeffs * y_slice)
    return torch.stack(result_slices, 1)


class PostMiniNetwork(nn.Module):
  def __init__(self, n_channels=[64,128]):
    super().__init__()
    self.conv1a = nn.Conv1d(in_channels=10, out_channels=n_channels[0], kernel_size=32)
    self.conv2a = nn.Conv1d(in_channels=n_channels[0], out_channels=n_channels[1], kernel_size=7)
    self.conv2b = nn.ConvTranspose1d(in_channels=n_channels[1], out_channels=n_channels[0], kernel_size=7)
    self.conv1b = nn.ConvTranspose1d(in_channels=n_channels[0], out_channels=10, kernel_size=32)

  def forward(self, x):
    mid0 = x
    mid1 = F.relu(self.conv1a(mid0))
    mid2 = F.relu(self.conv2a(mid1))
    x = mid2
    x = F.relu(self.conv2b(x)) + mid1
    x = F.relu(self.conv1b(x)) + mid0
    return x

class SepDiffConditionalModel(nn.Module):

  def __init__(self, Nsp, *args, **kwargs):
    super().__init__()
    self.Nsp = Nsp
    self.resampler_8k16k = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)
    self.resampler_8k22k = torchaudio.transforms.Resample(orig_freq=8000, new_freq=22050)
    self.resampler_16k22k = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050)
    self.resampler_22k16k = torchaudio.transforms.Resample(orig_freq=22050, new_freq=16000)
    self.separator = SepformerModel(Nsp)
    self.denoiser = DiffusionModel(Nsp)
    self.stft = lambda input : torch.stft(input.squeeze(-2), n_fft=1024, return_complex=True).unsqueeze(-3)  # sq-unsq C=1
    self.istft = lambda input : torch.istft(input.squeeze(-3), n_fft=1024, return_complex=False).unsqueeze(-2)
    self.mixer_angle = MixerModel(Nsp)
    self.mixer_abs = MixerModel(Nsp)
    self.dummy = nn.Parameter(torch.randn(1))
#    self.post_separator = nn.Sequential(
#      nn.Conv1d(in_channels=10, out_channels=64, kernel_size=32, padding='same'),
#      nn.ReLU(),
#      nn.Conv1d(in_channels=64, out_channels=10, kernel_size=7, padding='same'),
#    )
#    self.post_denoiser = nn.Sequential(
#      nn.Conv1d(in_channels=10, out_channels=64, kernel_size=32, padding='same'),
#      nn.ReLU(),
#      nn.Conv1d(in_channels=64, out_channels=10, kernel_size=7, padding='same'),
#    )
    self.post_separator = PostMiniNetwork()
    self.post_denoiser = PostMiniNetwork()
    # add conv-transpose, make U-Net
    # make it channel-independent, not n_channels=10

  def fix_length(self, tensor, L_target, lenience=10):
    B, C, L_old = tensor.shape
    # convs-deconvs may give, say, length 239864 instead of 239862, but we check it is not too much off:
    assert L_target - lenience <= L_old <= L_target + lenience, f'length before mixing is {L_old} but we need {L_target}, too much off'
    tensor = tensor[..., :L_target]
    tensor = F.pad(tensor, (0, L_target - tensor.shape[-1]))
    assert tensor.shape[-1] == L_target
    return tensor

  def forward(self, mixture_16k):
    B, C, L = mixture_16k.shape
    assert C == 1
    with torch.no_grad():
      separated_8k = self.separator(mixture_16k)
      separated_22k = self.resampler_8k22k(separated_8k)
      separated_16k = self.fix_length(self.resampler_22k16k(separated_22k), L, lenience=1)
      assert separated_16k.shape == (B, self.Nsp, L)
#    return dict(
#      **{f"separated{i+1}": separated_16k[:, i:i+1, :] for i in range(self.Nsp)},
#      **{f"predicted{i+1}": separated_16k[:, i:i+1, :] + 0 * self.dummy for i in range(self.Nsp)}
#    )
    with torch.no_grad():
      mixture_22k = self.resampler_16k22k(mixture_16k)
      denoised_22k = self.denoiser(separated_22k, mixture_22k)
      denoised_16k = self.resampler_22k16k(denoised_22k)
      denoised_16k = self.fix_length(denoised_16k, L, lenience=256)
      assert denoised_16k.shape == (B, self.Nsp, L)
    separated_16k_postbridge = separated_16k + self.post_separator(separated_16k)
    denoised_16k_postbridge = denoised_16k + self.post_denoiser(denoised_16k)
    separated = torch.cat([self.stft(separated_16k_postbridge[:, i, :]) for i in range(self.Nsp)], 1)
    denoised = torch.cat([self.stft(denoised_16k_postbridge[:, i, :]) for i in range(self.Nsp)], 1)
    predicted_angle = self.mixer_angle(separated.angle(), denoised.angle())
    predicted_abs = self.mixer_abs(separated.abs(), denoised.abs())
    predicted = torch.view_as_complex(predicted_abs.unsqueeze(-1) * torch.stack([torch.cos(predicted_angle), torch.sin(predicted_angle)], -1))
    predicted = torch.cat([self.istft(predicted[:, i, :, :]) for i in range(self.Nsp)], 1)
    return dict(
      **{f"separated{i+1}_prebridge": separated_16k[:, i:i+1, :] for i in range(self.Nsp)},
      **{f"denoised{i+1}_prebridge": denoised_16k[:, i:i+1, :] for i in range(self.Nsp)},
      **{f"separated{i+1}_postbridge": separated_16k_postbridge[:, i:i+1, :] for i in range(self.Nsp)},
      **{f"denoised{i+1}_postbridge": denoised_16k_postbridge[:, i:i+1, :] for i in range(self.Nsp)},
      **{f"predicted{i+1}": predicted[:, i:i+1, :] for i in range(self.Nsp)},
    )
