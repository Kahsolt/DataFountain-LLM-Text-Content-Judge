#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/08/31

# 选择题

from utils import *

CHOICES = ['A', 'B', 'C', 'D']

R_PUNCT_CN = Regex('[，。、…；：！？—（）【】《》‘’“”]+')
R_CHC_PRB = Regex('A、(.+) B、(.+) C、(.+) D、(.+)')
P_CHC_BLANK = '[，。；：！（）【】《》‘’“”,.;:!\(\)\[\]<>\"\'\*\s]*'
R_CHC_ANS = Regex('正确选项为([ABCD]) 模型回复为(.*)')
R_CHC_ANS_TMPL_LIST = [   # 粗过滤llm输出：先匹配文本，后匹配字母ABCD
  Regex(f'^{P_CHC_BLANK}([ABCD]){P_CHC_BLANK}$'),
  Regex(f'^选{P_CHC_BLANK}([ABCD])$'),
  Regex('answer is ?(.+)'),
  Regex('选项中?正确的是(.+)'),
  Regex('选项.*?(.+).*?正确'),
  Regex('推断出(.+)'),
  Regex('选择(.+)'),
  Regex('应该[为是]?(.+)'),
  Regex('回答[为是]?(.+)'),
  Regex('答案[为是]?(.+)'),
  Regex('([ABCD]、\S+)'),
  Regex('(.+)'),
]
R_CHC_ANS_TMPL_LIST_HIT = [0] * len(R_CHC_ANS_TMPL_LIST)
R_CHC_NOANS_TMPL_LIST = [
  Regex("don't know how to answer"),
  Regex("not safe or appropriate"),
  Regex('No content'),
  Regex('完全不想关选项或答案'),
  Regex('这个选项具有误导性'),
  Regex('我?还没有?学'),
  Regex('我?不能提供'),
  Regex('无法回答'),
  Regex('爱莫能助'),
  Regex('不支持'),
  Regex('抱歉'),
  Regex('全选'),
]
R_CHC_NOANS_TMPL_LIST_HIT = [0] * len(R_CHC_NOANS_TMPL_LIST)

letter_to_index = lambda c: ord(c) - ord('A')
index_to_letter = lambda i: chr(ord('A') + i)

P_TRIM = list('，。；：！（）【】《》‘’“”,.;:!()[]<>\"\'@#^&*')
def dequote(s:str) -> str:
  s = s.strip()
  while s[ 0] in P_TRIM: s = s[1:] .strip()
  while s[-1] in P_TRIM: s = s[:-1].strip()
  return s


# 按正则规则进行选择题的判断
# NOTE: 选择题须保证70%正确率才有提交得分
def judge_choices(quest:str, content:str, score:int=None) -> int:
  # match input & output tmpl.
  m = R_CHC_PRB.search(quest)
  sel_branches = m.groups()
  m = R_CHC_ANS.search(content)
  ref_ans_choice, llm_ans = m.groups()
  llm_ans_orig = llm_ans    # debug use
  llm_ans = dequote(llm_ans)
  ref_ans_text: str = dequote(sel_branches[letter_to_index(ref_ans_choice)])

  # test model rejection
  for tmpl_id, R in enumerate(R_CHC_NOANS_TMPL_LIST):
    if R.search(llm_ans):
      R_CHC_NOANS_TMPL_LIST_HIT[tmpl_id] += 1
      return 0

  # cleanify llm output, 排除定型文干扰字符
  llm_ans = llm_ans.replace('（内容由AI生成）', '')
  llm_ans = llm_ans.replace('AI', '')
  if '。解析：' in llm_ans: llm_ans = llm_ans[:llm_ans.index('。解析：')]
  #llm_ans = dequote(R_PUNCT_CN.sub('', llm_ans))
  llm_ans = dequote(llm_ans)

  # 尝试从滤过输出 llm_ans 中抽取 选项llm_ans_choice 和 描述llm_ans_text
  llm_ans_choice: str = None
  llm_ans_text: str = llm_ans
  if llm_ans[0] in CHOICES:      # 滤过输出是裸选项
    llm_ans_choice = llm_ans[0]
  else:
    for tmpl_id, R in enumerate(R_CHC_ANS_TMPL_LIST):
      m: re.Match = R.search(llm_ans)
      if m:
        R_CHC_ANS_TMPL_LIST_HIT[tmpl_id] += 1
        llm_ans_matched = dequote(m.groups()[0].strip())
        if llm_ans_matched in CHOICES:                  # 回答匹配到选项字母
          llm_ans_choice = llm_ans_matched
        else:                                           # 回答匹配到文本
          max_match_len = -1
          for chc, txt in zip(CHOICES, sel_branches):   # 回答文本反推选项字母
            match_len = len(txt)
            if dequote(txt) in llm_ans_matched and match_len > max_match_len:
              llm_ans_choice = chc
              max_match_len = match_len
          llm_ans_text = llm_ans_matched
        break

  # make judgement: choice deterministic!!
  if llm_ans_choice is not None:        # 回答选项 == 正确选项
    ok = ref_ans_choice == llm_ans_choice
  else:
    ok = ref_ans_text in llm_ans_text   # 回答文本 in 正确选支文本
    if not ok:                          # 回答选项 in 正确选支文本 (This is dangerous!)
      print(f'>> [DANGER]: |{ref_ans_choice}| == |{llm_ans_choice}| from {sel_branches}, or |{ref_ans_text}| in |{llm_ans_text}|')
      ok = ref_ans_choice in llm_ans_text
  judge = int(ok)

  if ref_ans_choice != llm_ans_choice and ref_ans_text in llm_ans_text:   # 可能出错的 case
    print(f'>> [ANTI]: |{ref_ans_choice}| == |{llm_ans_choice}| from {sel_branches}, or |{ref_ans_text}| in |{llm_ans_text}|')

  # verification-GT mismatch, we'll check these cases
  if score is not None and judge ^ score:
    print(f'[judge={judge}, GT={score}]', sel_branches, ref_ans_choice, llm_ans_choice or llm_ans_text or llm_ans)

  return judge


def judge_CHC(test_data:Dataset) -> Dataset:
  for segs in test_data:
    id, quest, dim, content, score = segs
    if dim == DIM_CHC:
      segs[-1] = judge_choices(quest, content, score)
  return test_data


def verify_ground_truth():
  # load data
  train_data = load_train_data()

  # analyze statis
  scores_CHC = []   # choices   (1049 samples)
  for segs in train_data:
    id, quest, dim, content, score = segs
    if dim == DIM_CHC:
      scores_CHC.append(score)
      judge_choices(quest, content, score)  # sanity check

  sc_CHC, p_CHC = samples_to_probdist(scores_CHC)
  print('p_CHC:', [round(e, 2) for e in p_CHC])


if __name__ == '__main__':
  verify_ground_truth()
  print('R_CHC_ANS_TMPL_LIST_HIT:', R_CHC_ANS_TMPL_LIST_HIT)
  print('R_CHC_NOANS_TMPL_LIST_HIT:', R_CHC_NOANS_TMPL_LIST_HIT)
