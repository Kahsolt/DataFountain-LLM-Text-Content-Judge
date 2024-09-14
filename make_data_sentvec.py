#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/14 

# 为 FLU/sentvec 任务准备训练数据

import pickle as pkl
from typing import Dict, Any

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils import *

DATA_TRAIN_FILE = OUT_PATH / 'train_data_sentvec.pkl'
DATA_TEST_B_FILE = OUT_PATH / 'test_b_data_sentvec.pkl'

DSample = Dict[str, Any]
DSamples = List[DSample]

# Run Configs ↓↓↓
if 'train':
  load_data = load_train_data
  save_fp = DATA_TRAIN_FILE
else:   # test-B
  load_data = load_test_B_data
  save_fp = DATA_TEST_B_FILE


# https://huggingface.co/spaces/mteb/leaderboard
SENT_VEC_MODEL = 'lier007/xiaobu-embedding-v2'    # 只用这个一个就行了


def make_sentvec(samples:DSamples):
  model = SentenceTransformer(SENT_VEC_MODEL, trust_remote_code=True, device=device)
  for idx, it in enumerate(tqdm(samples)):
    if it.get('sentvec') is not None: continue

    it['sentvec'] = model.encode([it['content']])[0]

    if (idx + 1) % 100 == 0:
      save_file(samples)

  del model
  save_file(samples)


def make_length(samples:DSamples):
  model_path = 'internlm/internlm2_5-1_8b-chat'
  tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
  )

  for idx, it in enumerate(tqdm(samples)):
    if it.get('len') is not None: continue

    it['len'] = len(tokenizer.encode(it['content']))

    if (idx + 1) % 100 == 0:
      save_file(samples)

  del tokenizer
  save_file(samples)


def load_file(fp:Path=None) -> DSamples:
  fp = fp or save_fp
  with open(fp, 'rb') as fh:
    return pkl.load(fh)


def save_file(samples:DSamples):
  with open(save_fp, 'wb') as fh:
    pkl.dump(samples, fh)


def run():
  data = load_data('FLU')
  print('n_samples:', len(data))

  if save_fp.exists():
    samples = load_file(save_fp)
  else:
    samples = [{
      'id':      e[0],
      'content': e[3],    # 大模型输出
      'score':   e[4],    # 人类评分
      'len':     None,    # 大模型输出 token 数
      'vec':     None,    # 大模型输出的句向量
    } for e in data]

  try:
    print('>> [Run] make_sentvec...')
    #make_sentvec(samples)

    print('>> [Run] make_length...')
    make_length(samples)
  finally:
    save_file(samples)


if __name__ == '__main__':
  run()
