#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/02 

# 为 FLU/ai_det 任务准备训练数据

import json
from typing import Dict, Any

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from utils import *

DATA_TRAIN_FILE = OUT_PATH / 'train_data_ai_det.json'

DSample = Dict[str, Any]
DSamples = List[DSample]


def make_ai_det_score(samples:DSamples):
  model_path = 'PirateXX/AI-Content-Detector'

  tokenizer = AutoTokenizer.from_pretrained(model_path)
  model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    device_map=device,
    #load_in_4bit=True,
    low_cpu_mem_usage=True,
  )

  def text_to_sentences(text:str) -> str:
    clean_text = text.replace('\n', ' ')
    return re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', clean_text)

  # function to concatenate sentences into chunks of size 900 or less
  def chunks_of_900(text, chunk_size = 900):
    sentences = text_to_sentences(text)
    chunks = []
    current_chunk = ''
    for sentence in sentences:
      if len(current_chunk + sentence) <= chunk_size:
        if len(current_chunk)!=0:
          current_chunk += " " + sentence
        else:
          current_chunk += sentence
      else:
        chunks.append(current_chunk)
        current_chunk = sentence
    chunks.append(current_chunk)
    return chunks

  def predict(query:str) -> float:
    tokens = tokenizer.encode(query)
    tokens = tokens[:tokenizer.model_max_length - 2]
    tokens = torch.tensor([tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]).unsqueeze(0)
    mask = torch.ones_like(tokens)

    with torch.inference_mode():
      logits = model(tokens.to(device), attention_mask=mask.to(device))[0]
      probs = logits.softmax(dim=-1)
    fake, real = probs.detach().cpu().flatten().numpy().tolist()
    return real

  def detect(text:str) -> float:
    chunksOfText = chunks_of_900(text)
    results = []
    for chunk in chunksOfText:
      output = predict(chunk)
      results.append([output, len(chunk)])
    ans, cnt = 0, 0
    for prob, length in results:
      cnt += length
      ans += prob * length
    realProb = ans / cnt
    return realProb

  for idx, it in enumerate(tqdm(samples)):
    if it['ai_det'] is not None: continue

    it['ai_det'] = detect(it['content'])

    if (idx + 1) % 10 == 0:
      save_file(samples)

  del model, tokenizer
  save_file(samples)


def load_file() -> DSamples:
  with open(DATA_TRAIN_FILE, encoding='utf-8') as fh:
    return json.load(fh)


def save_file(samples:DSamples):
  with open(DATA_TRAIN_FILE, 'w', encoding='utf-8') as fh:
    json.dump(samples, fh, indent=2, ensure_ascii=False)


def run():
  data = load_train_data('FLU')
  print('n_samples:', len(data))

  if DATA_TRAIN_FILE.exists():
    with open(DATA_TRAIN_FILE, encoding='utf-8') as fh:
      samples = json.load(fh)
  else:
    samples = [{
      'id':      e[0],
      'content': e[3],      # 大模型输出
      'score':   e[4],      # 人类评分
      'ai_det':  None,      # AI-Det 模型的评分
    } for e in data]

  try:
    print('>> [Run] make_ai_det_score...')
    make_ai_det_score(samples)
  finally:
    save_file(samples)


if __name__ == '__main__':
  run()
