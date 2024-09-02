#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/01 

# 流畅性

from utils import *


def judge_FLU_mean(test_data:Dataset) -> Dataset:
  for segs in test_data:
    id, quest, dim, content, score = segs
    if dim == DIM_FLU:
      segs[-1] = 3.1726666666666667
  return test_data


def judge_FLU_random(test_data:Dataset) -> Dataset:
  # load data
  train_data = load_train_data()
  # analyze statis
  scores_FLU = []   # fluency (1000 samples)
  for segs in train_data:
    id, quest, dim, content, score = segs
    if dim == DIM_FLU:
      scores_FLU.append(score)
  sc_FLU, p_FLU = samples_to_probdist(scores_FLU)
  print('mean(sc_FLU):', mean(scores_FLU))    # mean diff: 0.8579822222222222
  print('p_FLU:', [round(e, 2) for e in p_FLU])
  # random baseline solution
  for segs in test_data:
    id, quest, dim, content, score = segs
    if dim == DIM_FLU:
      segs[-1] = np.random.choice(sc_FLU, p=p_FLU)
  return test_data
