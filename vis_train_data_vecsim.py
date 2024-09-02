#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/02 

# 查看 make_data_vecsim 所预备的数据

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.base import RegressorMixin
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, mean_absolute_error
import matplotlib.pyplot as plt

from make_data_vecsim import *
from utils import *


''' Data '''
data = load_file()
data = [e for e in data if len(e['content']) > 0]
VECSIM_MODELS = sorted(data[0]['cossim'].keys())


''' Feature '''
X, Y = [], []
for it in data:
  sims: List[float] = []
  for metric in ['dotsim', 'cossim', 'aglsim']:
    for model in VECSIM_MODELS:
      sims.append(it[metric][model])
  X.append(sims)
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
  plt.tight_layout()
  plt.legend()
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
for model_cls in [
  lambda: KNeighborsClassifier(n_neighbors=3),
  KNeighborsClassifier,  # k=5
  LogisticRegression,
  BernoulliNB,
  GaussianNB,
  DecisionTreeClassifier,
  ExtraTreeClassifier,
  RandomForestClassifier,
  RandomForestRegressor,
  #AdaBoostClassifier,
  GradientBoostingClassifier,
]:
  is_rgr = issubclass(model_cls, RegressorMixin)

  print(f'[{model_cls.__name__}]')
  model = model_cls()
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  run_eval(y_test, y_pred, is_rgr)
