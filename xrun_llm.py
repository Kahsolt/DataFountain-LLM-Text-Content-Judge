#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/08/31 

# 交互式测试脚本: 跑LLM测原始权重在给定数据集上的输出

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2 import Qwen2ForCausalLM

model_path = 'internlm/internlm2_5-1_8b-chat'
max_new_tokens = 256

device = 'cuda:0'
model: Qwen2ForCausalLM = AutoModelForCausalLM.from_pretrained(
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


try:
  while True:
    prompt = input('>> input: ').strip()
    if not prompt: continue

    messages = [
      {"role": "system", "content": "你是一个文本摘要小助手。"},
      {"role": "user", "content": f'简短概括下述文本的主题大意，忽略细节内容，不超过140字：{prompt}'},
    ]
    text = tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True,
    )
    #print('>> processed text: ', text)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=max_new_tokens)
    generated_ids = [g[len(i):] for g, i in zip(generated_ids, model_inputs.input_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print('<<', response)
    print()
except KeyboardInterrupt:
  pass
