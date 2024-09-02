#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/02 

# 交互式测试脚本: 判断文本是否为AI生成

import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import device

# ref: https://huggingface.co/spaces/PirateXX/AI-Content-Detector
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


try:
  while True:
    prompt = input('>> input: ').strip()
    if not prompt: continue

    print('<<', detect(prompt))
    print()
except KeyboardInterrupt:
  pass
