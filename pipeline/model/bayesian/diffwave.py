import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet import linear
from speechbrain.nnet.diffusion import DenoisingDiffusion
from math import sqrt
from torchaudio import transforms


class DiffWaveDiffusionTuned(DenoisingDiffusion):
  """An enhanced diffusion implementation with DiffWave-specific inference
  Arguments
  ---------
  model: nn.Module
    the underlying model
  timesteps: int
    the total number of timesteps
  noise: str|nn.Module
    the type of noise being used
    "gaussian" will produce standard Gaussian noise
  beta_start: float
    the value of the "beta" parameter at the beginning of the process
    (see DiffWave paper)
  beta_end: float
    the value of the "beta" parameter at the end of the process
  show_progress: bool
    whether to show progress during inference
  """
  def __init__(
    self,
    model,
    timesteps=None,
    noise=None,
    beta_start=None,
    beta_end=None,
    sample_min=None,
    sample_max=None,
    show_progress=False,
  ):
    super().__init__(
      model,
      timesteps,
      noise,
      beta_start,
      beta_end,
      sample_min,
      sample_max,
      show_progress,
    )

  @torch.no_grad()
  def inference(
    self,
    unconditional,
    scale,
    conditions=None, initial_mixture_audio=None, initial_src_approximations=None,
    fast_sampling=False,
    fast_sampling_noise_schedule=None,
    device=None,
  ):
    assert conditions is not None and initial_mixture_audio is not None and initial_src_approximations is not None
    Nsp, _, _ = conditions.shape
    #print(f'inference called with {condition.shape=}')
    """Processes the inference for diffwave
    One inference function for all the locally/globally conditional
    generation and unconditional generation tasks
    Arguments
    ---------
    unconditional: bool
      do unconditional generation if True, else do conditional generation
    scale: int
      scale to get the final output wave length
      for conditional genration, the output wave length is scale * condition.shape[-1]
      for example, if the condition is spectrogram (bs, n_mel, time), scale should be hop length
      for unconditional generation, scale should be the desired audio length
    condition: torch.Tensor
      input spectrogram for vocoding or other conditions for other
      conditional generation, should be None for unconditional generation
    fast_sampling: bool
      whether to do fast sampling
    fast_sampling_noise_schedule: list
      the noise schedules used for fast sampling
    device: str|torch.device
      inference device
    Returns
    ---------
    predicted_sample: torch.Tensor
      the predicted audio (bs, 1, t)
    """
    if device is None:
      device = torch.device("cuda")
    # either condition or uncondition
    if unconditional:
      assert conditions is None
    else:
      assert conditions is not None
      device = conditions.device

    # must define fast_sampling_noise_schedule during fast sampling
    if fast_sampling:
      assert fast_sampling_noise_schedule is not None

    if fast_sampling and fast_sampling_noise_schedule is not None:
      inference_noise_schedule = fast_sampling_noise_schedule
      inference_alphas = 1 - torch.tensor(inference_noise_schedule)
      inference_alpha_cum = inference_alphas.cumprod(dim=0)
    else:
      inference_noise_schedule = self.betas
      inference_alphas = self.alphas
      inference_alpha_cum = self.alphas_cumprod

    inference_steps = []
    for s in range(len(inference_noise_schedule)):
      for t in range(self.timesteps - 1):
        if self.alphas_cumprod[t + 1] <= inference_alpha_cum[s] <= self.alphas_cumprod[t]:
          twiddle = (self.alphas_cumprod[t] ** 0.5 - inference_alpha_cum[s] ** 0.5) / (self.alphas_cumprod[t] ** 0.5 - self.alphas_cumprod[t + 1] ** 0.5)
          inference_steps.append(t + twiddle)
          break

    if not unconditional:
      if len(conditions.shape) == 3: # Expand rank 2 tensors by adding a batch dimension.
        conditions = conditions.unsqueeze(1)
      audios = torch.randn(Nsp, conditions.shape[1], scale * conditions.shape[-1], device=device)
    else:
      audios = torch.randn(Nsp, 1, scale, device=device)
    # noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)

    for n in range(len(inference_alphas) - 1, -1, -1):
      c1 = 1 / inference_alphas[n] ** 0.5
      c2 = (inference_noise_schedule[n] / (1 - inference_alpha_cum[n]) ** 0.5)
      # predict noise
      noises_pred = torch.stack([
        self.model(
          audio,
          torch.tensor([inference_steps[n]], device=device),
          condition,
        ).squeeze(1)
      for audio, condition in zip(audios, conditions)])
      # mean
      #print(f'{audios.shape=}, {noises_pred.shape=}, now shifting using {initial_mixture_audio.shape=}, {initial_src_approximations.shape=}')
      excess = audios.sum(0) - initial_mixture_audio[..., :audios.shape[-1]]
      discrepancy = audios - initial_src_approximations[..., :audios.shape[-1]]
      #print(f'{noises_pred=}, {excess=}, {discrepancy=}')
      audios = c1 * (audios - c2 * (noises_pred + 10.0 * excess + 0.0 * discrepancy))
      # add variance
      if n > 0:
        noises = torch.randn_like(audios)
        sigma = ((1.0 - inference_alpha_cum[n - 1]) / (1.0 - inference_alpha_cum[n]) * inference_noise_schedule[n]) ** 0.5
        audios += sigma * noises
      audios = torch.clamp(audios, -1.0, 1.0)
    return audios
