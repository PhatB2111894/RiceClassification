from scipy.io import arff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, classification_report
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
scaler = StandardScaler()
standard_x = scaler.fit_transform(X)

# train test split and inspect the shape
X_train, X_test, y_train, y_test = train_test_split(standard_x, Y, test_size=0.2, random_state=42, stratify=Y)
# print(f'train feature shape: {X_train.shape}')
# print(f'train target shape: {y_train.shape}')
# print(f'test feature shape: {X_test.shape}')
# print(f'test feature shape: {y_test.shape}')

# establish kfold
kf = KFold(n_splits=10, shuffle=True, random_state=42)
# support vector classification with parameter tuning on grid search
sv_param_grid = [
    {'C': [100], 'gamma': [0.1], 'kernel': ['sigmoid']}
]
svc_model = svm.SVC(probability=True)
svc_cv = GridSearchCV(svc_model, sv_param_grid, cv=kf, refit=True, verbose=0, scoring='accuracy')
svc_cv.fit(X_train, y_train)
print(f'Training Accuracy of SVC: {accuracy_score(svc_cv.predict(X_train), y_train)}')
print(f'Estimator for SVC: {svc_cv.best_estimator_}')
y_pred = svc_cv.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))