#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/08/31 

import csv
from re import compile as Regex
from typing import Tuple, List
from pathlib import Path

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

Sample = Tuple[str, str, str, str, int]   # (id, quest, judge_dim, content, score)
Dataset = List[Sample]

R_CHOICE_PRB = Regex('A、(.+) B、(.+) C、(.+) D、(.+)')
R_CHOICE_ANS = Regex('正确选项为([ABCD]) 模型回复为(.*)')


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
      assert dim in ['流畅性', '规范性', '选择题'], breakpoint()
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
      assert dim in ['流畅性', '规范性', '选择题'], breakpoint()
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


if __name__ == '__main__':
  from collections import Counter
  import numpy as np

  # load data
  train_data = load_train_data()
  print('len(train_data):', len(train_data))
  test_data = load_test_data()
  print('len(test_data):', len(test_data))

  def judge_choices(quest, content) -> Tuple[int, Tuple[List[str], str, str]]:
    # match input & output tmpl.
    m = R_CHOICE_PRB.search(quest)
    sel_branches = m.groups()
    m = R_CHOICE_ANS.search(content)
    ref_ans, llm_ans = m.groups()
    # cleanify llm output
    llm_ans_c = llm_ans.replace('AI', '')
    # make judgement
    found_k = ref_ans in llm_ans_c
    found_v = sel_branches[ord(ref_ans) - ord('A')] in llm_ans_c
    judge = int(found_k or found_v)
    return judge, (sel_branches, ref_ans, llm_ans)

  # analyze statis
  scores_FLU = []   # fluency   (1000 samples)
  scores_NOR = []   # normative (1000 samples)
  scores_CHC = []   # choices   (1049 samples)
  for segs in train_data:
    id, quest, dim, content, score = segs
    if   dim == '流畅性': scores_FLU.append(score)
    elif dim == '规范性': scores_NOR.append(score)
    elif dim == '选择题':
      scores_CHC.append(score)
      if 'sanity check':
        judge, (sel_branches, ref_ans, llm_ans) = judge_choices(quest, content)
        if judge ^ score:   # verification-GT mismatch, we'll check these cases
          print(f'[judge={judge}, GT={score}]', sel_branches, ref_ans, llm_ans)
    else: raise RuntimeError(f'unknown dim: {dim}')

  def samples_to_probdist(nums:List[int]) -> Tuple[List[int], List[float]]:
    cntr = Counter(nums)
    nums = sorted(cntr.keys())
    freq = [cntr[n] for n in nums]
    tot = sum(freq)
    prob = [e / tot for e in freq] 
    return nums, prob

  sc_FLU, p_FLU = samples_to_probdist(scores_FLU)
  sc_NOR, p_NOR = samples_to_probdist(scores_NOR)
  sc_CHC, p_CHC = samples_to_probdist(scores_CHC)
  print('p_FLU:', [round(e, 2) for e in p_FLU])
  print('p_NOR:', [round(e, 2) for e in p_NOR])
  print('p_CHC:', [round(e, 2) for e in p_CHC])

  # random baseline solution
  for segs in test_data:
    id, quest, dim, content, score = segs
    if   dim == '流畅性': segs[-1] = np.random.choice(sc_FLU, p=p_FLU)
    elif dim == '规范性': segs[-1] = np.random.choice(sc_NOR, p=p_NOR)
    elif dim == '选择题':
      # NOTE: 选择题须保证70%正确率才有提交得分
      #segs[-1] = np.random.choice(sc_CHC, p=p_CHC)
      judge, _ = judge_choices(quest, content)
      segs[-1] = judge
    else: raise RuntimeError(f'unknown dim: {dim}')

  # save file
  save_infer_data(test_data)
