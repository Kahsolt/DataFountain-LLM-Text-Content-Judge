#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/02 

# 重排原数据为json格式，方便人工观察规律

import json

from utils import *

TRAIN_FLU_FILE = OUT_PATH / 'train_FLU.json'
TRAIN_NOR_FILE = OUT_PATH / 'train_NOR.json'
TEST_FLU_FILE  = OUT_PATH / 'test_FLU.json'
TEST_NOR_FILE  = OUT_PATH / 'test_NOR.json'


data = load_train_data('FLU', ignore_empty_content=True)
data.sort(key=lambda e: (-e[-1], e[3], len(e[3])))
data = [(e[0], e[3], e[4]) for e in data]
with open(TRAIN_FLU_FILE, 'w', encoding='utf-8') as fh:
  json.dump(data, fh, indent=2, ensure_ascii=False)

data = load_train_data('NOR', ignore_empty_content=True)
data.sort(key=lambda e: (-e[-1], e[1], e[3], len(e[3])))
data = [(e[0], e[1], e[3], e[4]) for e in data]
with open(TRAIN_NOR_FILE, 'w', encoding='utf-8') as fh:
  json.dump(data, fh, indent=2, ensure_ascii=False)

data = load_test_data('FLU', ignore_empty_content=True)
data.sort(key=lambda e: (e[3], len(e[3])))
data = [(e[0], e[3]) for e in data]
with open(TEST_FLU_FILE, 'w', encoding='utf-8') as fh:
  json.dump(data, fh, indent=2, ensure_ascii=False)

data = load_test_data('NOR', ignore_empty_content=True)
data.sort(key=lambda e: (e[1], e[3], len(e[3])))
data = [(e[0], e[1], e[3]) for e in data]
with open(TEST_NOR_FILE, 'w', encoding='utf-8') as fh:
  json.dump(data, fh, indent=2, ensure_ascii=False)
