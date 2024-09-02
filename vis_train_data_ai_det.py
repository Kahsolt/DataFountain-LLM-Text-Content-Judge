#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/09/02 

# 查看 make_data_ai_det 所预备的数据

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Lasso, Lars
from sklearn.model_selection import train_test_split
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

from make_data_ai_det import *
from utils import *


''' Data '''
data = load_file()
data = [e for e in data if len(e['content']) > 0]


''' Feature '''
X, Y = [], []
for it in data:
  X.append((it['ai_det'],))
  Y.append(it['score'])

X = np.asarray(X, dtype=np.float32)
Y = np.asarray(Y, dtype=np.int8)
print('X.shape:', X.shape)
print('Y.shape:', Y.shape)


''' Visualize '''
if 'scatter':
  plt.scatter(X, Y)
  plt.tight_layout()
  plt.show()


''' Model '''
if 'rescale':
  scaler = StandardScaler()
  X = scaler.fit_transform(X)

def run_eval(y_test, y_pred):
  mae = mean_absolute_error(y_test, y_pred)
  print('mae:', mae)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
for model_cls in [
  lambda: KNeighborsRegressor(n_neighbors=3),
  KNeighborsRegressor,  # k=5
  LinearRegression,
  Lasso,
  Lars,
]:
  print(f'[{model_cls.__name__}]')
  model = model_cls()
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  run_eval(y_test, y_pred)
