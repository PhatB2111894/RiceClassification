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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

# Đọc dữ liệu
df = pd.read_excel('C:/Users/tuanp/OneDrive/Documents/GitHub/RiceClassification/Rice_Dataset_Commeo_and_Osmancik/Rice_Dataset_Commeo_and_Osmancik/Rice_Cammeo_Osmancik.xlsx')

# Tiền xử lý dữ liệu
X = df.drop('Class', axis=1).values
y = df['Class'].values

le = LabelEncoder()
y = le.fit_transform(y) # [0] Cammeo | [1] Osmancik

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Định nghĩa các tham số cho GridSearchCV
param_grid = {
    'criterion': ['entropy'],
    'max_depth': [5],
    'min_samples_split': [2],
    'min_samples_leaf': [2]
}

# Tạo GridSearchCV
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=5,
                           scoring='accuracy',
                           n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Lấy mô hình tốt nhất
best_model = grid_search.best_estimator_
print("Best parameters found by GridSearchCV:")
print(grid_search.best_params_)

# Đánh giá mô hình tốt nhất
y_pred = best_model.predict(X_test)

# Tính toán precision, recall và f1-score cho từng lớp
scorers = {
    'precision': make_scorer(precision_score, average=None), 
    'recall': make_scorer(recall_score, average=None),
    'f1_score': make_scorer(f1_score, average=None)
}

# Đánh giá mô hình tốt nhất với cross-validation
scores = cross_val_score(best_model, X, y, cv=5)
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

# Đánh giá mô hình tốt nhất với cross-validation
scores_precision = cross_val_score(best_model, X, y, cv=5, scoring='precision')
scores_recall = cross_val_score(best_model, X, y, cv=5, scoring='recall')
scores_f1 = cross_val_score(best_model, X, y, cv=5, scoring='f1')

# Lấy giá trị trung bình và độ lệch chuẩn của precision, recall và f1-score
precision_mean = scores_precision.mean()
recall_mean = scores_recall.mean()
f1_mean = scores_f1.mean()

precision_std = scores_precision.std()
recall_std = scores_recall.std()
f1_std = scores_f1.std()

# In precision, recall và f1-score
print("Precision: %0.4f (+/- %0.4f)" % (precision_mean, precision_std * 2))
print("Recall: %0.4f (+/- %0.4f)" % (recall_mean, recall_std * 2))
print("F1-score: %0.4f (+/- %0.4f)" % (f1_mean, f1_std * 2))


# Tính toán và in confusion matrix chuẩn hóa
conf_matrix = confusion_matrix(y_test, y_pred, normalize='true')
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Cammeo', 'Osmancik'], columns=['Cammeo', 'Osmancik'])
print(conf_matrix_df)
