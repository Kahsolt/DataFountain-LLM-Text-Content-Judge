#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/08/31 

import csv
import re
from re import compile as Regex
from pathlib import Path
from collections import Counter
from typing import Tuple, List

import torch
import numpy as np

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

Sample = Tuple[str, str, str, str, int]   # ([0]id, [1]quest, [2]judge_dim, [3]content, [4]score)
Dataset = List[Sample]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mean = lambda x: sum(x) / len(x) if len(x) else 0.0


def load_train_data(filter:str=None) -> Dataset:
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
  if filter is not None:
    dim_kanji = DIM_MAPPING[filter]
    samples = [e for e in samples if e[2] == dim_kanji]
  print('len(train_data):', len(samples))
  return samples

def load_test_data(filter:str=None) -> Dataset:
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
  if filter is not None:
    dim_kanji = DIM_MAPPING[filter]
    samples = [e for e in samples if e[2] == dim_kanji]
  print('len(test_data):', len(samples))
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


def samples_to_probdist(nums:List[int]) -> Tuple[List[int], List[float]]:
  cntr = Counter(nums)
  nums = sorted(cntr.keys())
  freq = [cntr[n] for n in nums]
  tot = sum(freq)
  prob = [e / tot for e in freq] 
  return nums, prob
