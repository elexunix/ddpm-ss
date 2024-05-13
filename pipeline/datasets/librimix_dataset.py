import json, os, logging, shutil, pathlib, random
import numpy as np, pandas as pd
from tqdm import tqdm
import torchaudio
from speechbrain.utils.data_utils import download_file

from pipeline.mixer.generate import generate
# Used to generate
#generate(all_audios_dir='data/datasets/librispeech/train-clean-360', out_folder='data/datasets/source-separation', n_files_train=30000, n_files_test=3000)  # change if needed
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


class LibriUnk3MixDataset(BaseDatasetSS):
  def __init__(self, libri0mix_root_dir,
                     libri1mix_clean_csv_filename, libri1mix_dirty_csv_filename,
                     libri2mix_clean_csv_filename, libri2mix_dirty_csv_filename,
                     libri3mix_clean_csv_filename, libri3mix_dirty_csv_filename,
               cnt_limit_0=None, cnt_limit_1=None, cnt_limit_2=None, cnt_limit_3=None, maxlen=None, *args, **kwargs):
    def extract_from_two_indexes(clean_csv_filename, dirty_csv_filename, cnt_limit=None):
      index = [row['mixture_path'] for i, row in pd.read_csv(clean_csv_filename).iterrows()] \
            + [row['mixture_path'] for i, row in pd.read_csv(dirty_csv_filename).iterrows()]
      random.shuffle(index)
      if cnt_limit is not None:
        index = index[:cnt_limit]
      return index
    self.index0 = []
    for root, _, filenames in os.walk(libri0mix_root_dir):
      for filename in filenames:
        if filename.endswith('.wav'):
          self.index0.append(os.path.join(root, filename))
    assert self.index0
    if cnt_limit_0 is not None:
      self.index0 = self.index0[:cnt_limit_0]
    self.index1 = extract_from_two_indexes(libri1mix_clean_csv_filename, libri1mix_dirty_csv_filename, cnt_limit_1)
    self.index2 = extract_from_two_indexes(libri2mix_clean_csv_filename, libri2mix_dirty_csv_filename, cnt_limit_2)
    self.index3 = extract_from_two_indexes(libri3mix_clean_csv_filename, libri3mix_dirty_csv_filename, cnt_limit_3)
    print('DATASET CONTAINS', len(self.index0), 'libri0mix audios',
                              len(self.index1), 'libri1mix audios',
                              len(self.index2), 'libri2mix audios',
                              len(self.index3), 'libri3mix audios')
    assert len(self.index0) == len(self.index1) == len(self.index2) == len(self.index3)
    self.index = [{'path_mixed': path, 'cnt_speakers': 0.0} for path in self.index1] \
               + [{'path_mixed': path, 'cnt_speakers': 1.0} for path in self.index1] \
               + [{'path_mixed': path, 'cnt_speakers': 2.0} for path in self.index2] \
               + [{'path_mixed': path, 'cnt_speakers': 3.0} for path in self.index3]
    random.shuffle(self.index)
    print(self.index[:5])
    super().__init__(self.index, *args, **kwargs)
    self.maxlen = maxlen

  def __getitem__(self, index):
    entry = self.index[index]
    return {
      'mixed': self.load_audio(entry['path_mixed'], maxlen=self.maxlen)[None],
      'cnt_speakers': np.float32(entry['cnt_speakers'])
    }

  def __len__(self):
    return len(self.index)
