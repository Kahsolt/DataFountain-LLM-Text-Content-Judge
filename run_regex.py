#!/usr/bin/env python3
# Author: Armit
# Create Time: 周六 2024/08/31

# 朴素的正则匹配法，仅处理选择题

import re
from re import compile as Regex
from collections import Counter
import numpy as np
from utils import *

R_CHOICE_PRB = Regex('A、(.+) B、(.+) C、(.+) D、(.+)')
R_CHOICE_ANS = Regex('正确选项为([ABCD]) 模型回复为(.*)')
R_CHOICE_ANS_TMPL_LIST = [
  # k
  Regex('([ABCD])\*'),
  Regex('\"{3}([ABCD])\"{3}'),
  Regex('answer.*?([ABCD])'),
  Regex('选项.*?([ABCD]).*?正确'),
  Regex('选.*?([ABCD]).*?项'),
  Regex('选择.*?([ABCD])'),
  Regex('.*?([ABCD]).*?项正确'),
  Regex('答案.*?([ABCD])'),
  Regex('回答.*?([ABCD])'),
  # v
  Regex('答案(.+)'),
  Regex('正确(.+)'),
  Regex('推断出(.+)'),
  Regex('选项(.+)正确'),
]
R_CHOICE_NOANS_TMPL_LIST = [
  Regex('No content'),
  Regex('抱歉'),
  Regex('无法回答'),
  Regex('不支持'),
  Regex('我?不能提供'),
  Regex('我?还没有?学'),
]


def run():
  # load data
  train_data = load_train_data()
  print('len(train_data):', len(train_data))
  test_data = load_test_data()
  print('len(test_data):', len(test_data))

  # 按正则规则进行选择题的判断
  letter_to_index = lambda c: ord(c) - ord('A')
  index_to_letter = lambda i: chr(ord('A') + i)
  def judge_choices(quest:str, content:str, score:int=None) -> Tuple[int, Tuple[List[str], str, str]]:
    # match input & output tmpl.
    m = R_CHOICE_PRB.search(quest)
    sel_branches = m.groups()
    m = R_CHOICE_ANS.search(content)
    ref_ans_choice, llm_ans = m.groups()
    llm_ans: str
    ref_ans_text: str = sel_branches[letter_to_index(ref_ans_choice)]

    # test model rejection
    for R in R_CHOICE_NOANS_TMPL_LIST:
      if R.search(llm_ans):
        return 0

    # cleanify llm output, 排除定型文干扰字符
    llm_ans = llm_ans.replace('（内容由AI生成）', '')
    # 尝试从完整回答 llm_ans 中抽取 选项llm_ans_choice 和 描述llm_ans_text
    llm_ans_choice: str = None
    llm_ans_text: str = llm_ans
    for R in R_CHOICE_ANS_TMPL_LIST:
      m: re.Match = R.search(llm_ans)
      if m:
        llm_ans_matched = m.groups()[0].strip()
        if llm_ans_matched in ['A', 'B', 'C', 'D']:
          llm_ans_choice = llm_ans_matched
        else:
          llm_ans_text = llm_ans_matched
        break

    # make judgement: 选项能对齐，或者
    ok: bool = False
    if not ok:    # 回答选项 == 正确选项
      ok = ref_ans_choice == llm_ans_choice
    if not ok:    # 回答文本 in 正确选支文本
      ok = ref_ans_text in llm_ans_text
    if not ok:    # 回答选项 in 正确选支文本 (dangerous)
      ok = ref_ans_choice in llm_ans_text
    judge = int(ok)

    # verification-GT mismatch, we'll check these cases
    if score is not None and judge ^ score:
      print(f'[judge={judge}, GT={score}]', sel_branches, ref_ans_choice, llm_ans_choice or llm_ans_text or llm_ans)

    return judge

  # analyze statis
  scores_FLU = []   # fluency   (1000 samples)
  scores_NOR = []   # normative (1000 samples)
  scores_CHC = []   # choices   (1049 samples)
  for segs in train_data:
    id, quest, dim, content, score = segs
    if   dim == DIM_FLU: scores_FLU.append(score)
    elif dim == DIM_NOR: scores_NOR.append(score)
    elif dim == DIM_CHC:
      scores_CHC.append(score)
      judge_choices(quest, content, score)  # sanity check
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
    if   dim == DIM_FLU: segs[-1] = np.random.choice(sc_FLU, p=p_FLU)
    elif dim == DIM_NOR: segs[-1] = np.random.choice(sc_NOR, p=p_NOR)
    elif dim == DIM_CHC:
      # NOTE: 选择题须保证70%正确率才有提交得分
      #segs[-1] = np.random.choice(sc_CHC, p=p_CHC)
      segs[-1] = judge_choices(quest, content)
    else: raise RuntimeError(f'unknown dim: {dim}')

  # save file
  save_infer_data(test_data)


if __name__ == '__main__':
  run()
