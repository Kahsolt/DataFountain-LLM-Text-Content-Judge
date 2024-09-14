#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/02 

# 查看 make_data_vecsim 所预备的数据

import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, mean_absolute_error
from ydata_profiling import ProfileReport
import pandas as pd
import matplotlib.pyplot as plt

from make_data_vecsim import *
from utils import *


''' Data '''
data = load_file()
data = [e for e in data if len(e['content']) > 0]
#VECSIM_MODELS = sorted(data[0]['cossim'].keys())
VECSIM_MODELS = ['sentence-transformers/paraphrase-multilingual-mpnet-base-v2']


''' Feature '''
X, Y = [], []
for it in data:
  sims: List[float] = [it['len']]
  for metric in ['dotsim', 'cossim', 'aglsim']:
    for model in VECSIM_MODELS:
      sims.append(it[metric][model])
  X.append(sims)
  Y.append(it['score'])

X = np.asarray(X, dtype=np.float32)
Y = np.asarray(Y, dtype=np.int8)
print('X.shape:', X.shape)
print('Y.shape:', Y.shape)


''' Data Profiling '''
fp = OUT_PATH / 'train_data_vecsim.html'
if not fp.exists():
  df = pd.DataFrame(X)
  ProfileReport(df).to_file(fp)
  print(f'>> save report to {fp}')


''' Visualize '''
if not 'pca':
  pca = PCA(n_components=3)
  X_pca = pca.fit_transform(X)
  print('explained_variance_:', pca.explained_variance_)
  print('explained_variance_ratio_:', pca.explained_variance_ratio_)

  ax = plt.subplot(projection='3d')
  ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=Y, cmap='prism')
  plt.suptitle('VecSim')
  plt.xlabel('vec sims')
  plt.ylabel('human score')
  plt.legend()
  plt.tight_layout()
  plt.show()


''' Model '''
if 'rescale':
  scaler = StandardScaler()
  X = scaler.fit_transform(X)

def run_eval(y_test, y_pred, is_rgr:bool=False):
  mae = mean_absolute_error(y_test, y_pred)
  print('mae:', mae)
  if is_rgr: y_pred = y_pred.round()
  prec, recall, f1, sup = precision_recall_fscore_support(y_test, y_pred)
  print(f'prec: {mean(prec):.3%}', prec)
  print(f'recall: {mean(recall):.3%}', recall)
  print(f'f1: {mean(f1):.3f}', f1)
  print(f'sup: {mean(sup):.3f}', sup)
  print(confusion_matrix(y_test, y_pred))
  print()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

if not 'try models':
  for model_cls in [
    lambda: KNeighborsClassifier(n_neighbors=3),
    KNeighborsClassifier,   # k=5
    KNeighborsRegressor,    # k=5
    LogisticRegression,
    BernoulliNB,
    GaussianNB,
    DecisionTreeClassifier,
    ExtraTreeClassifier,
    RandomForestClassifier,
    RandomForestRegressor,      # <- best!
    GradientBoostingClassifier,
  ]:
    is_rgr = model_cls in [RandomForestRegressor, KNeighborsRegressor]

    print(f'[{model_cls.__name__}]')
    model = model_cls()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    run_eval(y_test, y_pred, is_rgr)

'''
[随机摇奖摇出来的，别动！]
⚪ 1 个模型 dim=3
mae: 1.0730593894442653
prec: 48.942% [1.         0.14285714 0.36040609 0.1938326  0.75      ]
recall: 24.677% [0.02325581 0.04255319 0.5        0.64705882 0.02097902]
f1: 0.174 [0.04545455 0.06557377 0.41887906 0.29830508 0.04081633]
sup: 88.600 [ 43  47 142  68 143]
[[ 1  2 18 22  0]
 [ 0  2 28 17  0]
 [ 0  4 71 66  1]
 [ 0  1 23 44  0]
 [ 0  5 57 78  3]]
 ⚪ 1 个模型 + 长度特征 dim=4
mae: 0.8050793178479408
prec: 45.505% [0.79310345 0.13333333 0.48466258 0.18235294 0.68181818]
recall: 38.087% [0.53488372 0.04255319 0.55633803 0.45588235 0.31468531]
f1: 0.383 [0.63888889 0.06451613 0.51803279 0.2605042  0.43062201]
sup: 88.600 [ 43  47 142  68 143]
[[23 10  6  4  0]
 [ 2  2 21 18  4]
 [ 2  2 79 54  5]
 [ 0  0 25 31 12]
 [ 2  1 32 63 45]]
⚪ 9 个模型 dim=27
mae: 1.072441349971824
prec: 49.943% [1.         0.         0.34328358 0.15384615 1.        ]
recall: 21.982% [0.06976744 0.         0.48591549 0.52941176 0.01398601]
f1: 0.160 [0.13043478 0.         0.40233236 0.2384106  0.02758621]
sup: 88.600 [ 43  47 142  68 143]
[[ 3  1 18 21  0]
 [ 0  0 27 20  0]
 [ 0  2 69 71  0]
 [ 0  0 32 36  0]
 [ 0  0 55 86  2]]
'''
model = RandomForestRegressor(n_estimators=50, min_samples_leaf=2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
run_eval(y_test, y_pred, is_rgr=True)

if not 'save':
  fp = OUT_PATH / 'vecsim-rescaler.pkl'
  #fp = OUT_PATH / 'vecsim-rescaler.9.pkl'
  joblib.dump(scaler, fp)
  print(f'>> save model file to {fp}')
  fp = OUT_PATH / 'vecsim.pkl'
  #fp = OUT_PATH / 'vecsim.9.pkl'
  joblib.dump(model, fp)
  print(f'>> save model file to {fp}')
