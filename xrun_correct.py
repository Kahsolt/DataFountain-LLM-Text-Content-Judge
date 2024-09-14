#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/14

# 交互式测试脚本: 文本纠错
# - https://github.com/shibing624/pycorrector
# - https://github.com/HillZhang1999/MuCGEC

from pycorrector.t5.t5_corrector import T5Corrector
from pycorrector.macbert.macbert_corrector import MacBertCorrector


t5 = T5Corrector("shibing624/mengzi-t5-base-chinese-correction")
mb = MacBertCorrector("shibing624/macbert4csc-base-chinese")


def correct(text:str):
  return [
    t5(text),
    mb(text),
  ]

try:
  while True:
    prompt = input('>> input: ').strip()
    if not prompt: continue

    print('<<', correct(prompt))
    print()
except KeyboardInterrupt:
  pass
