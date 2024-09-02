#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/08/31 

# 查看输出长度对评分的影响
# - FLU: 无影响，均匀分布
# - NOR: 输出越长分数略越高

from argparse import ArgumentParser
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from utils import *


def run(args):
  model_path = 'internlm/internlm2_5-1_8b-chat'
  tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
  )

  data = load_train_data()
  data = [e for e in data if len(e[3])]

  dim = DIM_MAPPING[args.T]
  samples: Dataset = []
  for it in data:
    if dim == it[2]:
      samples.append(it)
  print('len(samples):', len(samples))

  lens   = [len(tokenizer.encode(it[3])) for it in samples]
  scores = [it[-1]     for it in samples]
  plt.scatter(lens, scores)
  plt.show()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-T', default='FLU', choices=['FLU', 'NOR', 'CHC'], help='sample type')
  args = parser.parse_args()

  run(args)
