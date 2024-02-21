import torch, torch.nn as nn, torch.nn.functional as F

from .diffusion import DiffusionModel
from .sepformer import SepformerModel


class SepDiffModel(nn.Module):

  def __init__(self, *args, **kwargs):
    super().__init__()
    self.separator = SepformerModel()
    self.denoiser = DiffusionModel()
    self.dummy = nn.Parameter(torch.randn(1))

  def fix_length(self, tensor, L_target):
    C, L_old = tensor.shape
    # convs-deconvs may give, say, length 239864 instead of 239862, but we check it is not too much off:
    assert L_target - 10 <= L_old <= L_target + 10
    tensor = tensor[:, :L_target]
    tensor = F.pad(tensor, (0, L_target - tensor.shape[-1]))
    assert tensor.shape[-1] == L_target
    return tensor

  def forward(self, mixture):
    with torch.no_grad():
      separated1, separated2 = self.separator(mixture)
    separated1 = self.denoiser(separated1) + 0 * self.dummy
    separated2 = self.denoiser(separated2)
    L = mixture.shape[-1]
    separated1 = self.fix_length(separated1, L)
    separated2 = self.fix_length(separated2, L)
    denoised1 = self.denoiser(separated1)
    denoised2 = self.denoiser(separated2)
    #return {"predicted1": predicted1, "predicted2": predicted2}
    return {"predicted1": denoised1, "predicted2": denoised2}
