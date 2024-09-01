#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/08/31 

# 交互式测试脚本: 跑LLM测原始权重在给定数据集上的输出

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2 import Qwen2ForCausalLM

# https://huggingface.co/Qwen/Qwen1.5-7B-Chat
model_path = 'Qwen/Qwen1.5-7B-Chat'
max_new_tokens = 512

device = 'cuda:0'
model: Qwen2ForCausalLM = AutoModelForCausalLM.from_pretrained(
  model_path,
  #torch_dtype=torch.bfloat16, 
  device_map=device,
  load_in_4bit=True,
  low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)


try:
  while True:
    prompt = input('>> input: ').strip()
    if not prompt: continue

    messages = [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True,
    )
    #print('>> processed text: ', text)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=max_new_tokens)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print('<<', response)
    print()
except KeyboardInterrupt:
  pass
