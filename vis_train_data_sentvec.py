#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/14 

# 查看 make_data_sentvec 所预备的数据

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
import matplotlib.pyplot as plt

from make_data_sentvec import *
from utils import *


''' Data '''
data = load_file()
data = [e for e in data if len(e['content']) > 0]

''' Feature '''
X, Y = [], []
for it in data:
  feats = np.pad(it['sentvec'], (1, 0), constant_values=it['len']) 
  X.append(feats)
  Y.append(it['score'])

X = np.asarray(X, dtype=np.float32)
Y = np.asarray(Y, dtype=np.int8)
print('X.shape:', X.shape)
print('Y.shape:', Y.shape)


''' Visualize '''
if not 'pca':
  pca = PCA(n_components=3)
  X_pca = pca.fit_transform(X)
  print('explained_variance_:', pca.explained_variance_)
  print('explained_variance_ratio_:', pca.explained_variance_ratio_)

  ax = plt.subplot(projection='3d')
  ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=Y, cmap='prism')
  plt.suptitle('SentVec')
  plt.xlabel('sent vec')
  plt.ylabel('human score')
  plt.legend()
  plt.tight_layout()
  plt.show()


''' Model '''
if 'rescale':
  scaler = StandardScaler()
  X = scaler.fit_transform(X)

# D=1793 必须降维!
pca = PCA(n_components=100)
X = pca.fit_transform(X)

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
    #lambda: KNeighborsClassifier(n_neighbors=3),
    #KNeighborsClassifier,   # k=5
    #KNeighborsRegressor,    # k=5
    #LogisticRegression,
    #BernoulliNB,
    #GaussianNB,
    #DecisionTreeClassifier,
    #ExtraTreeClassifier,
    RandomForestClassifier,   # <- best!
    #RandomForestRegressor,
    #GradientBoostingClassifier,
  ]:
    is_rgr = model_cls in [RandomForestRegressor, KNeighborsRegressor]

    print(f'[{model_cls.__name__}]')
    model = model_cls()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    run_eval(y_test, y_pred, is_rgr)

'''
[随机摇奖摇出来的，别动！]
mae: 0.8177777777777778
prec: 14.365% [0.         0.         0.43255814 0.28571429 0.        ]
recall: 20.319% [0.         0.         0.97382199 0.04210526 0.        ]
f1: 0.134 [0.         0.         0.59903382 0.0733945  0.        ]
sup: 90.000 [ 50  60 191  95  54]
[[  0   1  47   2   0]
 [  0   0  58   1   1]
 [  0   0 186   5   0]
 [  1   0  87   4   3]
 [  0   0  52   2   0]]
'''
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
run_eval(y_test, y_pred, is_rgr=True)

if not 'save':
  fp = OUT_PATH / 'sentvec-rescaler.pkl'
  joblib.dump(scaler, fp)
  print(f'>> save model file to {fp}')
  fp = OUT_PATH / 'sentvec-pca.pkl'
  joblib.dump(pca, fp)
  print(f'>> save model file to {fp}')
  fp = OUT_PATH / 'sentvec.pkl'
  joblib.dump(model, fp)
  print(f'>> save model file to {fp}')
