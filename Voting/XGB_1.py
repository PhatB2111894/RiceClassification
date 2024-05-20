from scipy.io import arff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
import os

data = arff.loadarff('C:/Users/tuanp/OneDrive/Documents/GitHub/RiceClassification/Rice_Dataset_Commeo_and_Osmancik/Rice_Dataset_Commeo_and_Osmancik/Rice_Cammeo_Osmancik.arff')
df = pd.DataFrame(data[0])
df['Class'] = df['Class'].str.decode('utf-8')
df.head()

df.shape
df.isnull().sum()
df.describe()

df['Class'].value_counts(normalize=True)

# Split for features and targets, standardize features
X = df.iloc[:,:-1].to_numpy()
Y = df.iloc[:,-1].astype('category').cat.codes.to_numpy()
scaler = MinMaxScaler()
standard_x = scaler.fit_transform(X)

# train test split and inspect the shape
X_train, X_test, y_train, y_test = train_test_split(standard_x, Y, test_size=0.2, random_state=42, stratify=Y)

# establish kfold
kf = KFold(n_splits=10, shuffle=True, random_state=42)

xgb_params = {'n_estimators':np.arange(100,1000,100),
              'max_depth':np.arange(3,10),
              'gamma':np.arange(0,0.5,0.1),
              'reg_alpha':[0,0.0001,0.001,0.01,0.1]}
xgb = XGBClassifier(learning_rate=0.5,random_state=42)
xgb_cv = GridSearchCV(xgb, xgb_params, cv=kf, refit=True, scoring='accuracy',verbose=0)
xgb_cv.fit(X_train, y_train)
print(f'Training Accuracy of XGB: {accuracy_score(xgb_cv.predict(X_train), y_train)}')
print(f'Estimator for XGB: {xgb_cv.best_estimator_}')

# Đánh giá độ chính xác của mô hình tốt nhất trên tập test
best_xgb_model = xgb_cv.best_estimator_
y_pred_test = best_xgb_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f'Test Accuracy of the best XGBoost model: {test_accuracy}')
