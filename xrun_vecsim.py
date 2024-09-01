#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/03

# 交互式测试脚本: 模型输出文本的相似度

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pairwise_dot_score, pairwise_cos_sim, pairwise_angle_sim

from utils import device

PRETRAINED_CHECKPOINTS = [
  # official: https://www.sbert.net/docs/pretrained_models.html#model-overview
  'all-mpnet-base-v2',
  'all-distilroberta-v1',
  'all-MiniLM-L6-v2',
  'all-MiniLM-L12-v2',
  'multi-qa-mpnet-base-dot-v1',
  'multi-qa-distilbert-cos-v1',
  'multi-qa-MiniLM-L6-cos-v1',
  'paraphrase-albert-small-v2',
  'paraphrase-MiniLM-L3-v2',
  'paraphrase-multilingual-mpnet-base-v2',
  'paraphrase-multilingual-MiniLM-L12-v2',
  'distiluse-base-multilingual-cased-v1',
  'distiluse-base-multilingual-cased-v2',
  # contrib
  'amu/tao-k',
  'BAAI/bge-m3',
  'aspire/acge_text_embedding',
  'mixedbread-ai/mxbai-embed-large-v1',
]

model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
model = SentenceTransformer(model_name, device=device)

try:
  while True:
    s0 = input('>> input task description: ').strip()
    if s0 == 'q': break
    if s0 == 'r': 
      print('=' * 72)
      continue
    s1 = input('>> input model output: ').strip()
    if s0 == 'q': break
    if s0 == 'r':
      print('=' * 72)
      continue

    embed_adv, embed_ref = model.encode([s0, s1], convert_to_tensor=True)
    embed_adv.unsqueeze_(0)
    embed_ref.unsqueeze_(0)
    dot_score = pairwise_dot_score(embed_adv, embed_ref).item()
    cos_sim   = pairwise_cos_sim  (embed_adv, embed_ref).item()
    angle_sim = pairwise_angle_sim(embed_adv, embed_ref).item()

    print(f'dot_score: {dot_score}')
    print(f'cos_sim: {cos_sim}')
    print(f'angle_sim: {angle_sim}')

except KeyboardInterrupt:
  pass
