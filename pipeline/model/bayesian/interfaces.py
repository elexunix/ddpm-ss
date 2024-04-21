import torch
from speechbrain.pretrained.interfaces import Pretrained
from .diffwave import DiffWaveDiffusionTuned


class DiffWaveConditionalInferer(Pretrained):
  """
  A ready-to-use inference wrapper for DiffWave as vocoder.
  The wrapper allows to perform generative tasks:
    locally-conditional generation: mel_spec -> waveform
  Arguments
  ---------
  hparams
    Hyperparameters (from HyperPyYAML)
  """

  HPARAMS_NEEDED = ["diffusion"]

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if hasattr(self.hparams, "diffwave"):
      self.infer = self.hparams.diffusion.inference
      #self.infer = DiffWaveDiffusionTuned.inference
    else:
      raise NotImplementedError

  def decode_batch(
    self,
    mels_batch, m0s_batch, xs_batch,
    hop_len,
    mel_lens=None,
    fast_sampling=False,
    fast_sampling_noise_schedule=None,
  ):
    """Generate waveforms from spectrograms
    Arguments
    ---------
    mel: torch.tensor
      spectrogram [batch, mels, time]
    hop_len: int
      Hop length during mel-spectrogram extraction
      Should be the same value as in the .yaml file
      Used to determine the output wave length
      Also used to mask the noise for vocoding task
    mel_lens: torch.tensor
      Used to mask the noise caused by padding
      A list of lengths of mel-spectrograms for the batch
      Can be obtained from the output of Tacotron/FastSpeech
    fast_sampling: bool
      whether to do fast sampling
    fast_sampling_noise_schedule: list
      the noise schedules used for fast sampling
    Returns
    -------
    waveforms: torch.tensor
      Batch of mel-waveforms [batch, 1, time]

    """
    #print(f'decode_batch: {mels_batch.shape=}, {m0s_batch.shape=}, {xs_batch.shape=}')
    # decode_batch: mels_batch.shape=torch.Size([8, 2, 80, 353]), m0s_batch.shape=torch.Size([8, 1, 90317]), xs_batch.shape=torch.Size([8, 2, 90368]) 
    with torch.no_grad():
      waveforms = torch.stack([
        self.infer(
          unconditional=False,
          scale=hop_len,
          conditions=mels_of_audio.to(self.device),
          initial_mixture_audio=m0,
          initial_src_approximations=xs0[:, None, :],
          fast_sampling=fast_sampling,
          fast_sampling_noise_schedule=fast_sampling_noise_schedule,
        )
      for mels_of_audio, m0, xs0 in zip(mels_batch, m0s_batch, xs_batch)])
      assert waveforms.shape[-2] == 1
      waveforms = waveforms[:, :, 0, :]
    # Mask the noise caused by padding during batch inference
    if mel_lens is not None and hop_len is not None:
      assert False
      waveform = self.mask_noise(waveform, mel_lens, hop_len)
    return waveforms
