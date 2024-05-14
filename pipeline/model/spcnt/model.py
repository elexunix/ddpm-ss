import torch, torch.nn as nn

class SpeakerCountingModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.stack = nn.Sequential(
      nn.Conv1d(in_channels=1, out_channels=128, kernel_size=1024, stride=256),
      nn.ReLU(),
      nn.Conv1d(in_channels=128, out_channels=256, kernel_size=32, stride=4),
      nn.ReLU(),
      nn.Conv1d(in_channels=256, out_channels=1, kernel_size=16)
    )
    self.dummy = nn.Parameter(torch.randn(1))
  def forward(self, x):
    B, C, L = x.shape
    assert C == 1
    x = x * 30 / x.norm(dim=-1, keepdim=True)
    #return torch.full((B,), 1.5, dtype=torch.float, device=x.device) + 0 * self.dummy  # baseline
    x = self.stack(x).mean(-1)
    assert x.shape == (B, 1)
    return x.squeeze(1)
  def kekward(self, x):
    B, C, L = x.shape
    assert C == 1
    x = x * 30 / x.norm(dim=-1, keepdim=True)
    #return torch.full((B,), 1.5, dtype=torch.float, device=x.device) + 0 * self.dummy  # baseline
    x = self.stack(x)
    l = x.shape[-1]
    assert x.shape == (B, 1, l)
    # and we want these guys to be at most 1:
    speakers_aggregated_excess = (x - 1)[x > 1].sum()
    assert speakers_aggregated_excess.ndim == 0
    return speakers_aggregated_excess
