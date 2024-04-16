from scipy.io import arff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
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

# Split for features and targets, standardize features
X = df.iloc[:,:-1].to_numpy()
Y = df.iloc[:,-1].astype('category').cat.codes.to_numpy()
scaler = MinMaxScaler()
standard_x = scaler.fit_transform(X)

# train test split and inspect the shape
X_train, X_test, y_train, y_test = train_test_split(standard_x, Y, test_size=1/3, random_state=42, stratify=Y)

# establish kfold
kf = KFold(n_splits=10, shuffle=True, random_state=42)

xgb_params = {'n_estimators': [100],
              'max_depth': [3],
              'gamma': [0.4],
              'reg_alpha': [0.01]}
xgb = XGBClassifier(learning_rate=0.1,random_state=42)
xgb_cv = GridSearchCV(xgb, xgb_params, cv=kf, refit=True, scoring='accuracy',verbose=0)
xgb_cv.fit(X_train, y_train)
print(f'Training Accuracy of XGB: {accuracy_score(xgb_cv.predict(X_train), y_train)}')
print(f'Estimator for XGB: {xgb_cv.best_estimator_}')
# Dự đoán nhãn cho tập dữ liệu kiểm tra
y_pred_test = xgb_cv.predict(X_test)

# Báo cáo lớp (classification report)
print("Classification Report:")
print(classification_report(y_test, y_pred_test,digits=4))

# Ma trận nhầm lẫn (confusion matrix)
cm = confusion_matrix(y_test, y_pred_test)
print("Confusion Matrix:")
print(cm)

# Hiển thị ma trận nhầm lẫn dưới dạng heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=xgb_cv.classes_, yticklabels=xgb_cv.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()