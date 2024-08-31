#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/08/31 

import csv
from pathlib import Path
from typing import Tuple, List

BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / 'data'
OUT_PATH = BASE_PATH / 'out'
TRAIN_DATA_FILE = DATA_PATH / 'train.csv'   # n_samples: 3500 = 1500 + 1500 + 500
TEST_DATA_FILE  = DATA_PATH / 'test.csv'    # n_samples: 3049 = 1000 + 1000 + 1049
DEFAULT_OUT_FILE = OUT_PATH / 'submit.csv'
FILE_ENCODING = 'gb18030'                   # gb2312 < gbk < gb18030

'''
- 流畅性：对于文章的语句通顺性、语法错误、字词错误、表述方式进行评分 (1 ~ 5 分)
  - 1分: 非常不流畅
  - 5分: 非常流畅
- 规范性：对于文章的内容遵循性、格式规范性、语句逻辑、层次结构进行评分 (1 ~ 5 分)
  - 1分: 创作内容离题，与提示语句要求不符，格式非常不规范
  - 5分: 创作内容与提示语句要求完美契合，格式非常规范
'''

DIM_FLU = '流畅性'
DIM_NOR = '规范性'
DIM_CHC = '选择题'
DIM_LIST = [DIM_FLU, DIM_NOR, DIM_CHC]
DIM_MAPPING = {
  'FLU': '流畅性',
  'NOR': '规范性', 
  'CHC': '选择题',
}

Sample = Tuple[str, str, str, str, int]   # (id, quest, judge_dim, content, score)
Dataset = List[Sample]


def load_train_data() -> Dataset:
  with open(TRAIN_DATA_FILE, encoding=FILE_ENCODING) as fh:
    samples: Dataset = []
    is_first = True
    for segs in csv.reader(fh):
      if is_first:
        is_first = False
        continue
      assert len(segs) == 5, breakpoint()
      id, quest, dim, content, score = segs
      assert dim in DIM_LIST, breakpoint()
      samples.append((id, quest, dim, content, int(score)))
  return samples

def load_test_data() -> Dataset:
  with open(TEST_DATA_FILE, encoding=FILE_ENCODING) as fh:
    samples: Dataset = []
    is_first = True
    for segs in csv.reader(fh):
      if is_first:
        is_first = False
        continue
      assert len(segs) == 4, breakpoint()
      id, quest, dim, content = segs
      assert dim in DIM_LIST, breakpoint()
      samples.append([id, quest, dim, content, None])   # use list here to allow inplace-modify :)
  return samples

def save_infer_data(samples:Dataset, fp:Path=None):
  fp = fp or DEFAULT_OUT_FILE
  fp.parent.mkdir(exist_ok=True)

  data = [
    ['数据编号', '评判维度', '预测分数'],
  ]
  for segs in samples:
    id, quest, dim, content, score = segs
    data.append([id, dim, score])
  print(f'>> write csv: {fp}')
  with open(fp, 'w', encoding=FILE_ENCODING, newline='') as fh:
    writer = csv.writer(fh)
    writer.writerows(data)
