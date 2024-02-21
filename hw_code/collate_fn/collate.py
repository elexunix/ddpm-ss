import logging
from typing import List
import torch, torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
  """
  Collate and pad fields in dataset items
  """
  collated_batch = {}
  text_enc_lens, spec_lens = [], []
  for key, value0 in dataset_items[0].items():
    values = [item[key] for item in dataset_items]
    if isinstance(value0, torch.Tensor):
      lens = [value.shape[-1] for value in values]
      if key == 'text_encoded':
        text_enc_lens = lens
      elif key == 'spectrogram':
        spec_lens = lens
      collated_batch[key] = torch.vstack([F.pad(v, (0, max(lens) - v.shape[-1])) for v in values])
      device = value0.device
    else:
      collated_batch[key] = default_collate(values)
  collated_batch['text_encoded_length'] = torch.tensor(text_enc_lens, device=device)
  collated_batch['spectrogram_length'] = torch.tensor(spec_lens, device=device)
  #for k, v in collated_batch.items():
  #  print('key', k, 'value/shape', v.shape if isinstance(v, torch.Tensor) else v)

  return collated_batch
