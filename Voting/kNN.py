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

# fig, ax = plt.subplots(2,4, figsize=(25,25))
# fig.delaxes(ax[1,3])
# for ax, feature in zip(ax.flat, df.columns[:-1]):
#     plot = sns.histplot(data=df,x=feature,hue='Class', ax=ax, alpha=0.6).set_title(feature)

# Split for features and targets, standardize features
X = df.iloc[:,:-1].to_numpy()
Y = df.iloc[:,-1].astype('category').cat.codes.to_numpy()
scaler = MinMaxScaler()
standard_x = scaler.fit_transform(X)

# train test split and inspect the shape
X_train, X_test, y_train, y_test = train_test_split(standard_x, Y, test_size=1/3, random_state=42, stratify=Y)
# print(f'train feature shape: {X_train.shape}')
# print(f'train target shape: {y_train.shape}')
# print(f'test feature shape: {X_test.shape}')
# print(f'test feature shape: {y_test.shape}')

# establish kfold
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# knn with parameter tuning on grid search
knn_params = {'n_neighbors': np.arange(1,10),
              'weights': ['uniform', 'distance'],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1, 2]}
knn_model = KNeighborsClassifier()
knn_cv = GridSearchCV(knn_model, knn_params, cv=kf, verbose=0, scoring='accuracy', refit=True)
knn_cv.fit(X_train, y_train)
print(f'Training Accuracy of KNN: {knn_cv.best_score_}')
print(f'Estimator of KNN: {knn_cv.best_estimator_}')