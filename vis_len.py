#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/08/31 

# 查看输出长度对评分的影响 (结论是没啥影响...)

from argparse import ArgumentParser
import matplotlib.pyplot as plt
from utils import *


def run(args):
  data = load_train_data()

  dim = DIM_MAPPING[args.T]
  samples: Dataset = []
  for it in data:
    if dim == it[2]:
      samples.append(it)
  print('len(samples):', len(samples))

  lens   = [len(it[3]) for it in samples]
  scores = [it[-1]     for it in samples]
  plt.scatter(lens, scores)
  plt.show()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-T', default='FLU', choices=['FLU', 'NOR', 'CHC'], help='sample type')
  args = parser.parse_args()

  run(args)
