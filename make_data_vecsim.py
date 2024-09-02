#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/01 

# 为 NOR/vecsim 任务准备训练数据

import json
from typing import Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pairwise_dot_score, pairwise_cos_sim, pairwise_angle_sim
from tqdm import tqdm

from judge_NOR import R_NOR_TASK
from utils import *

DATA_TRAIN_FILE = OUT_PATH / 'train_data_vecsim.json'

DSample = Dict[str, Any]
DSamples = List[DSample]


# https://huggingface.co/spaces/mteb/leaderboard
EMBEDDING_MODEL_NAMES = [
  'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',    # 1.11GB
  'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',    # 471MB
  # ↓↓↓ chinese models listed by rank
  'lier007/xiaobu-embedding-v2',              # 1.21GB
  'iampanda/zpoint_large_embedding_zh',       # 1.21GB
  'Classical/Yinka',                          # 0.61GB
  'aspire/acge_text_embedding',               # 1.21GB
  'dunzhang/stella-mrl-large-zh-v3.5-1792d',  # 1.21GB
  'infgrad/stella-base-zh-v3-1792d',          # 0.38GB
  #'Amu/tao-8k',                               # 0.62GB (not available)
  'jinaai/jina-embeddings-v2-base-zh',        # 322MB
]


def make_summary_text(samples:DSamples):
  model_path = 'internlm/internlm2_5-1_8b-chat'
  max_new_tokens = 256

  device = 'cuda:0'
  model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    #torch_dtype=torch.bfloat16, 
    device_map=device,
    load_in_4bit=True,
    low_cpu_mem_usage=True,
  )
  tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
  )

  def model_infer(prompt:str) -> str:
    messages = [
      {"role": "system", "content": "你是一个文本摘要小助手。"},
      {"role": "user", "content": f'简短概括下述文本的主题大意，忽略细节内容，不超过140字：{prompt}'},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=max_new_tokens)
    generated_ids = [g[len(i):] for g, i in zip(generated_ids, model_inputs.input_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

  for idx, it in enumerate(tqdm(samples)):
    if it['content_skel']: continue

    it['content_skel'] = model_infer(it['content'])

    if (idx + 1) % 10 == 0:
      save_file(samples)

  del model, tokenizer
  save_file(samples)


def make_vecsim_socre(samples:DSamples):
  for model_name in tqdm(EMBEDDING_MODEL_NAMES):
    model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    for idx, it in enumerate(tqdm(samples)):
      if all(model_name in it[e] for e in ['dotsim', 'cossim', 'aglsim']): continue

      embed_ref = model.encode([it['quest']],        convert_to_tensor=True)
      embed_out = model.encode([it['content_skel']], convert_to_tensor=True)
      it['dotsim'][model_name] = pairwise_dot_score(embed_out, embed_ref).item()
      it['cossim'][model_name] = pairwise_cos_sim  (embed_out, embed_ref).item()
      it['aglsim'][model_name] = pairwise_angle_sim(embed_out, embed_ref).item()

      if (idx + 1) % 10 == 0:
        save_file(samples)

    del model
    save_file(samples)
  save_file(samples)


def load_file() -> DSamples:
  with open(DATA_TRAIN_FILE, encoding='utf-8') as fh:
    return json.load(fh)


def save_file(samples:DSamples):
  with open(DATA_TRAIN_FILE, 'w', encoding='utf-8') as fh:
    json.dump(samples, fh, indent=2, ensure_ascii=False)


def run():
  data = load_train_data('NOR')
  print('n_samples:', len(data))

  if DATA_TRAIN_FILE.exists():
    with open(DATA_TRAIN_FILE, encoding='utf-8') as fh:
      samples = json.load(fh)
  else:
    samples = [{
      'id':      e[0],
      'quest':   R_NOR_TASK.search(e[1]).groups()[0],   # 原题目要求
      'content': e[3],          # 大模型输出
      'score':   e[4],          # 人类评分
      'content_skel':  None,    # 大模型输出
      'dotsim':  {},            # 各句向量模型对 quest 和 content_skel 的相似度打分
      'cossim':  {},
      'aglsim':  {},
    } for e in data]

  try:
    print('>> [Run] make_summary_text...')
    make_summary_text(samples)

    print('>> [Run] make_vecsim_socre...')
    make_vecsim_socre(samples)
  finally:
    save_file(samples)


if __name__ == '__main__':
  run()
