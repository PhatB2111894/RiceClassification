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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
df = pd.read_excel('C:/Users/tuanp/OneDrive/Documents/GitHub/RiceClassification/Rice_Dataset_Commeo_and_Osmancik/Rice_Dataset_Commeo_and_Osmancik/Rice_Cammeo_Osmancik.xlsx')
df.shape
df.info()
df.describe()
df['Class'].unique()
df['Class'].value_counts()
df.isnull().sum().sort_values(ascending=False)
df.isna().sum().sort_values(ascending=False)


rice_class_0 = df[df['Class'] == 'Cammeo']
rice_class_1 = df[df['Class'] == 'Osmancik']


X = df.drop('Class', axis=1).values
y = df['Class'].values

le = LabelEncoder()
y = le.fit_transform(y) # [0] Cammeo | [1] Osmancik

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = DecisionTreeClassifier(random_state=42)
# Khởi tạo và huấn luyện mô hình Bagging Decision Tree
bagging_dt_model = BaggingClassifier(estimator=model, n_estimators=10, random_state=0)
bagging_dt_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred_bagging_dt = bagging_dt_model.predict(X_test)

# Calculate and print accuracy using cross-validation
scores = cross_val_score(model, X, y, cv=5)
print(scores)
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

# Calculate and print precision, recall, and F1-score using cross-validation
scorers = {
    'precision': make_scorer(precision_score, average=None),
    'recall': make_scorer(recall_score, average=None),
    'f1_score': make_scorer(f1_score, average=None)
}

scores_precision = cross_val_score(model, X, y, cv=5, scoring='precision')
scores_recall = cross_val_score(model, X, y, cv=5, scoring='recall')
scores_f1 = cross_val_score(model, X, y, cv=5, scoring='f1')

precision_mean = scores_precision.mean()
recall_mean = scores_recall.mean()
f1_mean = scores_f1.mean()

precision_std = scores_precision.std()
recall_std = scores_recall.std()
f1_std = scores_f1.std()

print("Precision: %0.4f (+/- %0.4f)" % (precision_mean, precision_std * 2))
print("Recall: %0.4f (+/- %0.4f)" % (recall_mean, recall_std * 2))
print("F1-score: %0.4f (+/- %0.4f)" % (f1_mean, f1_std * 2))

# Calculate normalized confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_bagging_dt, normalize='true')

# Save normalized confusion matrix to a DataFrame
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Cammeo', 'Osmancik'], columns=['Cammeo', 'Osmancik'])
print(conf_matrix_df)