import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
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


class SepformerSeparation10(Pretrained):
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
  def __init__(self, sepformer3_ckpt_state_dict, *args, **kwargs):
    super().__init__(*args, **kwargs)
    encoder, masknet, decoder = self.mods['encoder'], self.mods['masknet'], self.mods['decoder']
    #to_be_freezed = [encoder, decoder]
    #to_be_freezed += [masknet.conv1d, masknet.dual_mdl, masknet.output, masknet.output_gate]
    to_be_trained = [masknet.conv2d, masknet.end_conv1x1]
    #print([type(tbf) for tbf in to_be_freezed])
    init_weights_dict = sepformer3_ckpt_state_dict
    #for name in init_weights_dict:
    #  print('init has', name)
    for name, p in self.named_parameters():
      p.requires_grad = False
      if name in init_weights_dict:
        #print('trying to init', name, 'from pretrained...', end=' ')
        if p.shape == init_weights_dict[name].shape:
          p.copy_(init_weights_dict[name])
          #p.requires_grad = False
          p.requires_grad = True
          #print('success!')
        else:
          print('shape mismatched for', name, 'so this is one of the guys to train')
          p.requires_grad = True
      else:
        print('ERROR', name, 'not found in pretrained model from which we init!')
        quit(57)
    print('init from pretrained completed!')
    for name, p in self.named_parameters():
      if p.requires_grad:
        print('training parameter', name, 'shape', p.shape)
    cnt_trained = sum(np.prod(p.shape) for p in self.parameters() if p.requires_grad)
    print('Total number of trainable parameters:', cnt_trained)
    #assert cnt_trained == 328960

  def separate_batch(self, mix):
    #print(f'SepformerSeparation10 got {mix.shape=}')
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
    #print(f'{mix.shape=}, {est_source.shape=}')
    T_origin = mix.size(1)
    T_est = est_source.size(1)
    assert abs(T_origin - T_est) <= 256
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
