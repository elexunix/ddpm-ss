import torch, torch.nn as nn
from types import SimpleNamespace
#from speechbrain.pretrained import Pretrained


class Pretrained(nn.Module):
  HPARAMS_NEEDED = []
  def __init__(self, modules, hparams):
    super().__init__()
    self.mods = nn.ModuleDict(modules)
    self.device = 'cuda'
    for module in self.mods.values():
      if module is not None:
        module.to(self.device)
    super().__init__()
    # Arguments passed via the run opts dictionary. Set a limited
    # number of these, since some don't apply to inference.
    run_opts = None
    run_opt_defaults = {
      "device": "cuda",  # changed from cpu
      "data_parallel_count": -1,
      "data_parallel_backend": False,
      "distributed_launch": False,
      "distributed_backend": "nccl",
      "jit": False,
      "jit_module_keys": None,
      "compile": False,
      "compile_module_keys": None,
      "compile_mode": "reduce-overhead",
      "compile_using_fullgraph": False,
      "compile_using_dynamic_shape_tracing": False,
    }
    for arg, default in run_opt_defaults.items():
      setattr(self, arg,
        run_opts[arg] if run_opts is not None and arg in run_opts else
        # If any arg from run_opt_defaults exist in hparams and not in command line args "run_opts"
        hparams[arg] if hparams is not None and arg in hparams else
        default
      )

    # Put modules on the right device, accessible with dot notation
    self.mods = torch.nn.ModuleDict(modules)
    for module in self.mods.values():
      if module is not None:
        module.to(self.device)

    # Check MODULES_NEEDED and HPARAMS_NEEDED and
    # make hyperparams available with dot notation
    if self.HPARAMS_NEEDED and hparams is None:
      raise ValueError("Need to provide hparams dict.")
    if hparams is not None:
      # Also first check that all required params are found:
      for hp in self.HPARAMS_NEEDED:
        if hp not in hparams:
          raise ValueError(f"Need hparams['{hp}']")
      self.hparams = SimpleNamespace(**hparams)
    # Prepare modules for computation, e.g. jit
    #self._prepare_modules(freeze_params)


class SepformerSeparation5(Pretrained):
  """A "ready-to-use" speech separation model.
  Uses Sepformer architecture.
  Example
  -------
  >>> tmpdir = getfixture("tmpdir")
  >>> model = SepformerSeparation.from_hparams(
  ...   source="speechbrain/sepformer-wsj02mix",
  ...   savedir=tmpdir)
  >>> mix = torch.randn(1, 400)
  >>> est_sources = model.separate_batch(mix)
  >>> print(est_sources.shape)
  torch.Size([1, 400, 2])
  """
  MODULES_NEEDED = ["encoder", "masknet", "decoder"]

  def separate_batch(self, mix):
    #print(f'SepformerSeparation5 got {mix.shape=}')
    """Run source separation on batch of audio.
    Arguments
    ---------
    mix : torch.Tensor
      The mixture of sources.

    Returns
    -------
    tensor
      Separated sources
    """
    # Separation
    mix = mix.to(self.device)
    mix_w = self.mods.encoder(mix)
    est_mask = self.mods.masknet(mix_w)
    mix_w = torch.stack([mix_w] * self.hparams.num_spks)
    sep_h = mix_w * est_mask
    # Decoding
    est_source = torch.cat([
      self.mods.decoder(sep_h[i]).unsqueeze(-1)
      for i in range(self.hparams.num_spks)
    ],-1)
    # T changed after conv1d in encoder, fix it here
    T_origin = mix.size(1)
    T_est = est_source.size(1)
    if T_origin > T_est:
      est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
    else:
      est_source = est_source[:, :T_origin, :]
    return est_source

  def forward(self, mix):
    """Runs separation on the input mix"""
    B, C, L = mix.shape
    assert C == 1
    mix = mix.squeeze(1)
    separated = self.separate_batch(mix)
    #print(f'{separated.shape=}')
    return separated
