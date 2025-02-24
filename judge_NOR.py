#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/01 

# 规范性

from typing import Dict, Any

import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from utils import *

R_NOR_TASK = Regex('请以“(.+)”为主题写一篇文章')


def judge_NOR_mean(test_data:Dataset) -> Dataset:
  for segs in test_data:
    id, quest, dim, content, score = segs
    if dim == DIM_NOR:
      if len(content) > 0:
        segs[-1] = 3.509333333333333
      else:   # fix special case (data missing)
        segs[-1] = 5
  return test_data


def judge_NOR_random(test_data:Dataset) -> Dataset:
  # load data
  train_data = load_train_data()
  # analyze statis
  scores_NOR = []   # normative (1000 samples)
  for segs in train_data:
    id, quest, dim, content, score = segs
    if dim == DIM_NOR:
      scores_NOR.append(score)
  sc_NOR, p_NOR = samples_to_probdist(scores_NOR)
  print('mean(sc_NOR):', mean(scores_NOR))    # mean diff: 1.1282986666666668
  print('p_NOR:', [round(e, 2) for e in p_NOR])
  # random baseline solution
  for segs in test_data:
    id, quest, dim, content, score = segs
    if dim == DIM_NOR:
      segs[-1] = np.random.choice(sc_NOR, p=p_NOR)
  return test_data


scaler: StandardScaler = None
model: RandomForestRegressor = None
dsamples: List[Dict[str, Any]] = None

VECSIM_MODELS = ['sentence-transformers/paraphrase-multilingual-mpnet-base-v2']

def _load_env():    # load pretrained model & preprocessed data
  from make_data_vecsim import load_file
  global scaler, model, dsamples
  if scaler is None:
    scaler = joblib.load(OUT_PATH / 'vecsim-rescaler.pkl')
  if model is None:
    model = joblib.load(OUT_PATH / 'vecsim.pkl')
  if dsamples is None:
    dsamples = load_file(OUT_PATH / 'test_b_data_vecsim.json')

def _infer(it:Dict[str, Any]) -> float:
  sims: List[float] = [it['len']]
  for metric in ['dotsim', 'cossim', 'aglsim']:
    for m in VECSIM_MODELS:
      sims.append(it[metric][m])
  X = np.asarray([sims], dtype=np.float32)
  X = scaler.transform(X)
  pred = model.predict(X).item()
  return pred

def judge_NOR(test_data:Dataset) -> Dataset:
  _load_env()   # load data
  for segs in test_data:
    id, quest, dim, content, score = segs
    if dim == DIM_NOR:
      it = next(filter(lambda e: e['id'] == id, dsamples))
      y_pred = _infer(it)
      sc = round(y_pred, 7)
      segs[-1] = max(1, min(5, sc))
  return test_data


def verify_ground_truth():
  # load data
  train_data = load_train_data()
  test_data = load_test_data()

  # format check
  for segs in train_data:
    id, quest, dim, content, score = segs
    if dim == DIM_NOR:
      assert R_NOR_TASK.search(quest) 
  for segs in test_data:
    id, quest, dim, content, score = segs
    if dim == DIM_NOR:
      assert R_NOR_TASK.search(quest)


if __name__ == '__main__':
  verify_ground_truth()
