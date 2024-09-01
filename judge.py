#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/01 

# 主运行脚本: 读取测试数据集 -> 判分 -> 保存提交文件

from judge_FLU import judge_FLU_dummy
from judge_NOR import judge_NOR_dummy
from judge_CHC import judge_CHC

from utils import *


def run():
  # load data
  test_data = load_test_data()

  # make judgements, devide & conquer
  test_data = judge_FLU_dummy(test_data)
  test_data = judge_NOR_dummy(test_data)
  test_data = judge_CHC(test_data)

  # save file
  save_infer_data(test_data)


if __name__ == '__main__':
  run()
