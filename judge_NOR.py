#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/01 

# 规范性

from utils import *

R_NOR_TASK = Regex('请以“(.+)”为主题写一篇文章')


def judge_NOR_dummy(test_data:Dataset) -> Dataset:
  # load data
  train_data = load_train_data()
  # analyze statis
  scores_NOR = []   # normative (1000 samples)
  for segs in train_data:
    id, quest, dim, content, score = segs
    if dim == DIM_NOR:
      scores_NOR.append(score)
  sc_NOR, p_NOR = samples_to_probdist(scores_NOR)
  print('p_NOR:', [round(e, 2) for e in p_NOR])
  # random baseline solution
  for segs in test_data:
    id, quest, dim, content, score = segs
    if dim == DIM_NOR:
      segs[-1] = np.random.choice(sc_NOR, p=p_NOR)
  return test_data


def verify_ground_truth():
  # load data
  train_data = load_train_data()
  test_data = load_test_data()

  # format check
  for segs in train_data:
    id, quest, dim, content, score = segs
    if dim == DIM_NOR:
      assert R_NOR_TASK.search(quest) 
  for segs in test_data:
    id, quest, dim, content, score = segs
    if dim == DIM_NOR:
      assert R_NOR_TASK.search(quest)


if __name__ == '__main__':
  verify_ground_truth()
