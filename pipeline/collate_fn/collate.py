import logging
from typing import List
import torch, torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

logger = logging.getLogger(__name__)


def generic_collate(dataset_items: List[dict]):
  """
  Collate torch.Tensors along their last dim
  """
  collated_batch = {}
  text_enc_lens, spec_lens = [], []
  for key, value0 in dataset_items[0].items():
    values = [item[key] for item in dataset_items]
    if isinstance(value0, torch.Tensor):
      lens = [value.shape[-1] for value in values]
      collated_batch[key] = torch.vstack([F.pad(v, (0, max(lens) - v.shape[-1])) for v in values])
    else:
      collated_batch[key] = default_collate(values)
  return collated_batch


def collate_fn_asr(dataset_items: List[dict]):
  """
  Collate and pad fields in dataset items
  """
  collated_batch = generic_collate(dataset_items)
  for key, value0 in dataset_items[0].items():
    values = [item[key] for item in dataset_items]
    if key == 'text_encoded':
      text_enc_lens = [value.shape[-1] for value in values]
      device = value0.device
    elif key == 'spectrogram':
      spec_lens = [value.shape[-1] for value in values]
  collated_batch['text_encoded_length'] = torch.tensor(text_enc_lens, device=device)
  collated_batch['spectrogram_length'] = torch.tensor(spec_lens, device=device)
  return collated_batch


def collate_fn_ddpmss(dataset_items: List[dict]):
  """
  Again, collate and pad fields in dataset items \o_O/
  """
  collated_batch = generic_collate(dataset_items)
  return collated_batch


collate_fn = collate_fn_ddpmss
