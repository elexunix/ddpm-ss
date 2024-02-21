import torch
from torch import Tensor
from torch.nn import CTCLoss


class CTCLossWrapper(CTCLoss):
  def forward(self, log_probs, log_probs_length, text_encoded, text_encoded_length,
        **batch) -> Tensor:
    log_probs_t = torch.transpose(log_probs, 0, 1)
    #print(f'{log_probs_t.shape=}, {text_encoded.shape=}, {log_probs_length=}, {text_encoded_length=}')

    return super().forward(
      log_probs=log_probs_t,
      targets=text_encoded,
      input_lengths=log_probs_length,
      target_lengths=text_encoded_length,
    )
