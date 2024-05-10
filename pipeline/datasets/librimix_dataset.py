from pipeline.mixer.generate import generate
# Used to generate
#generate(all_audios_dir='data/datasets/librispeech/train-clean-360', out_folder='data/datasets/source-separation', n_files_train=30000, n_files_test=3000)  # change if needed


import json, os, logging, shutil, pathlib, pandas as pd
import torchaudio
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

from pipeline.base.base_dataset_ss import BaseDatasetSS
from pipeline.utils import ROOT_PATH

logger = logging.getLogger(__name__)


class Libri2MixDataset(BaseDatasetSS):
  def __init__(self, csv_filename, cnt_limit=None, maxlen=None, *args, **kwargs):
    #assert part in ['train', 'test']
    #if data_dir is None:
    #  data_dir = ROOT_PATH / 'data' / 'datasets' / 'source-separation' / part
    #self.data_dir = data_dir
    self.index = []
    for index, row in pd.read_csv(csv_filename).iterrows():
      self.index.append({
        "path_mixed": row["mixture_path"],
        "path_target1": row["source_1_path"],
        "path_target2": row["source_2_path"],
      })
      if cnt_limit is not None and index >= cnt_limit - 1:
        break
    super().__init__(self.index, *args, **kwargs)
    self.maxlen = maxlen

  def __getitem__(self, index):
    return {
      'mixed': self.load_audio(self.index[index]['path_mixed'], maxlen=self.maxlen)[None],
      'target1': self.load_audio(self.index[index]['path_target1'], maxlen=self.maxlen)[None],
      'target2': self.load_audio(self.index[index]['path_target2'], maxlen=self.maxlen)[None],
    }

  def __len__(self):
    return len(self.index)


class Libri3MixDataset(BaseDatasetSS):
  def __init__(self, csv_filename, cnt_limit=None, maxlen=None, *args, **kwargs):
    #self.data_dir = data_dir
    self.index = []
    for index, row in pd.read_csv(csv_filename).iterrows():
      self.index.append({
        "path_mixed": row["mixture_path"],
        "path_target1": row["source_1_path"],
        "path_target2": row["source_2_path"],
        "path_target3": row["source_3_path"],
      })
      if cnt_limit is not None and index >= cnt_limit - 1:
        break
    super().__init__(self.index, *args, **kwargs)
    self.maxlen = maxlen

  def __getitem__(self, index):
    return {
      'mixed': self.load_audio(self.index[index]['path_mixed'], maxlen=self.maxlen)[None],
      'target1': self.load_audio(self.index[index]['path_target1'], maxlen=self.maxlen)[None],
      'target2': self.load_audio(self.index[index]['path_target2'], maxlen=self.maxlen)[None],
      'target3': self.load_audio(self.index[index]['path_target3'], maxlen=self.maxlen)[None],
    }

  def __len__(self):
    return len(self.index)


class Libri5MixDataset(BaseDatasetSS):
  def __init__(self, csv_filename, cnt_limit=None, maxlen=None, *args, **kwargs):
    self.index = []
    for index, row in pd.read_csv(csv_filename).iterrows():
      self.index.append({
        "path_mixed": row["mixture_path"],
        "path_target1": row["source_1_path"],
        "path_target2": row["source_2_path"],
        "path_target3": row["source_3_path"],
        "path_target4": row["source_4_path"],
        "path_target5": row["source_5_path"],
      })
      if cnt_limit is not None and index >= cnt_limit - 1:
        break
    super().__init__(self.index, *args, **kwargs)
    self.maxlen = maxlen

  def __getitem__(self, index):
    return {
      'mixed': self.load_audio(self.index[index]['path_mixed'], maxlen=self.maxlen)[None],
      'target1': self.load_audio(self.index[index]['path_target1'], maxlen=self.maxlen)[None],
      'target2': self.load_audio(self.index[index]['path_target2'], maxlen=self.maxlen)[None],
      'target3': self.load_audio(self.index[index]['path_target3'], maxlen=self.maxlen)[None],
      'target4': self.load_audio(self.index[index]['path_target4'], maxlen=self.maxlen)[None],
      'target5': self.load_audio(self.index[index]['path_target5'], maxlen=self.maxlen)[None],
    }

  def __len__(self):
    return len(self.index)


class Libri10MixDataset(BaseDatasetSS):
  def __init__(self, csv_filename, cnt_limit=None, maxlen=None, *args, **kwargs):
    self.index = []
    for index, row in pd.read_csv(csv_filename).iterrows():
      self.index.append({
        "path_mixed": row["mixture_path"],
        "path_target1": row["source_1_path"],
        "path_target2": row["source_2_path"],
        "path_target3": row["source_3_path"],
        "path_target4": row["source_4_path"],
        "path_target5": row["source_5_path"],
        "path_target6": row["source_6_path"],
        "path_target7": row["source_7_path"],
        "path_target8": row["source_8_path"],
        "path_target9": row["source_9_path"],
        "path_target10": row["source_10_path"],
      })
      if cnt_limit is not None and index >= cnt_limit - 1:
        break
    super().__init__(self.index, *args, **kwargs)
    self.maxlen = maxlen

  def __getitem__(self, index):
    return {
      'mixed': self.load_audio(self.index[index]['path_mixed'], maxlen=self.maxlen)[None],
      'target1': self.load_audio(self.index[index]['path_target1'], maxlen=self.maxlen)[None],
      'target2': self.load_audio(self.index[index]['path_target2'], maxlen=self.maxlen)[None],
      'target3': self.load_audio(self.index[index]['path_target3'], maxlen=self.maxlen)[None],
      'target4': self.load_audio(self.index[index]['path_target4'], maxlen=self.maxlen)[None],
      'target5': self.load_audio(self.index[index]['path_target5'], maxlen=self.maxlen)[None],
      'target6': self.load_audio(self.index[index]['path_target6'], maxlen=self.maxlen)[None],
      'target7': self.load_audio(self.index[index]['path_target7'], maxlen=self.maxlen)[None],
      'target8': self.load_audio(self.index[index]['path_target8'], maxlen=self.maxlen)[None],
      'target9': self.load_audio(self.index[index]['path_target9'], maxlen=self.maxlen)[None],
      'target10': self.load_audio(self.index[index]['path_target10'], maxlen=self.maxlen)[None],
    }

  def __len__(self):
    return len(self.index)
