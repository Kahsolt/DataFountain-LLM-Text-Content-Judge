#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/01 

# 流畅性

from typing import Dict, Any

import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from utils import *


def judge_FLU_mean(test_data:Dataset) -> Dataset:
  for segs in test_data:
    id, quest, dim, content, score = segs
    if dim == DIM_FLU:
      segs[-1] = 3.1726666666666667
  return test_data


def judge_FLU_random(test_data:Dataset) -> Dataset:
  # load data
  train_data = load_train_data()
  # analyze statis
  scores_FLU = []   # fluency (1000 samples)
  for segs in train_data:
    id, quest, dim, content, score = segs
    if dim == DIM_FLU:
      scores_FLU.append(score)
  sc_FLU, p_FLU = samples_to_probdist(scores_FLU)
  print('mean(sc_FLU):', mean(scores_FLU))    # mean diff: 0.8579822222222222
  print('p_FLU:', [round(e, 2) for e in p_FLU])
  # random baseline solution
  for segs in test_data:
    id, quest, dim, content, score = segs
    if dim == DIM_FLU:
      segs[-1] = np.random.choice(sc_FLU, p=p_FLU)
  return test_data


scaler: StandardScaler = None
pca: PCA = None
model: RandomForestClassifier = None
dsamples: List[Dict[str, Any]] = None

def _load_env():    # load pretrained model & preprocessed data
  from make_data_sentvec import load_file
  global scaler, pca, model, dsamples
  if scaler is None:
    scaler = joblib.load(OUT_PATH / 'sentvec-rescaler.pkl')
  if pca is None:
    pca = joblib.load(OUT_PATH / 'sentvec-pca.pkl')
  if model is None:
    model = joblib.load(OUT_PATH / 'sentvec.pkl')
  if dsamples is None:
    dsamples = load_file(OUT_PATH / 'test_b_data_sentvec.pkl')

def _infer(it:Dict[str, Any]) -> float:
  X = np.pad(it['sentvec'], (1, 0), constant_values=it['len'])
  X = np.expand_dims(X, axis=0)
  X = scaler.transform(X)
  X = pca.transform(X)
  pred = model.predict(X).item()
  return pred

def judge_FLU(test_data:Dataset) -> Dataset:
  _load_env()   # load data
  for segs in test_data:
    id, quest, dim, content, score = segs
    if dim == DIM_FLU:
      it = next(filter(lambda e: e['id'] == id, dsamples))
      y_pred = _infer(it)
      sc = round(y_pred, 7)
      segs[-1] = max(1, min(5, sc))
  return test_data
