import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy.stats as stats

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import cross_val_score

df = pd.read_excel('C:/Users/tuanp/OneDrive/Documents/GitHub/RiceClassification/Rice_Dataset_Commeo_and_Osmancik/Rice_Dataset_Commeo_and_Osmancik/Rice_Cammeo_Osmancik.xlsx')
df.shape
df.info()
df.describe()
df['Class'].unique()
df['Class'].value_counts()
df.isnull().sum().sort_values(ascending=False)
df.isna().sum().sort_values(ascending=False)

var = 'Class'
status_counts = df[var].value_counts()

rice_class_0 = df[df['Class'] == 'Cammeo']
rice_class_1 = df[df['Class'] == 'Osmancik']

X = df[['Major_Axis_Length', 'Eccentricity']].values
y = df['Class'].values

le = LabelEncoder()
y = le.fit_transform(y) # [0] Cammeo | [1] Osmancik

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
n = math.ceil(math.sqrt(df.shape[0]))
print(f"N = {n}")
model = KNeighborsClassifier(n_neighbors=n)

bagging_knn_model = BaggingClassifier(estimator=model, n_estimators=10, random_state=0)
bagging_knn_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_bagging_knn = bagging_knn_model.predict(X_test)

# In ra báo cáo phân loại
print("Bagging kNN:")
print(classification_report(y_test, y_pred_bagging_knn, digits=4))

matrix = confusion_matrix(y_test, y_pred_bagging_knn)
display_labels = ['Cammeo', 'Osmancik']

mc_visual = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=display_labels)
mc_visual.plot()
scores = cross_val_score(bagging_knn_model, X, y, cv=5)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))