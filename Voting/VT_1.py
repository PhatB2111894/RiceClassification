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
X_train, X_test, y_train, y_test = train_test_split(standard_x, Y, test_size=0.2, random_state=42, stratify=Y)
# print(f'train feature shape: {X_train.shape}')
# print(f'train target shape: {y_train.shape}')
# print(f'test feature shape: {X_test.shape}')
# print(f'test feature shape: {y_test.shape}')

# establish kfold
kf = KFold(n_splits=10, shuffle=True, random_state=42)
# support vector classification with parameter tuning on grid search
sv_param_grid = {'C': [100, 1000, 10000, 10000],
                 'gamma': [0.1, 1, 10],
                 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
svc_model = svm.SVC(probability=True)
svc_cv = GridSearchCV(svc_model, sv_param_grid, cv=kf, refit=True, verbose=0, scoring='accuracy')
svc_cv.fit(X_train, y_train)
print(f'Training Accuracy of SVC: {accuracy_score(svc_cv.predict(X_train), y_train)}')
print(f'Estimator for SVC: {svc_cv.best_estimator_}')


# train_acc = {'Support Vector': accuracy_score(svc_cv.predict(X_train), y_train),
#              'XGBoost': accuracy_score(xgb_cv.predict(X_train),y_train),
#              'KNN': accuracy_score(knn_cv.predict(X_train),y_train)}
# plt.bar(range(len(train_acc)),train_acc.values())
# plt.xticks(range(len(train_acc)),list(train_acc.keys()))
# plt.title('Training Accuracy Comparison')
# plt.show()

# estimator = []
# estimator.append(('SVC', svc_cv.best_estimator_))
# estimator.append(('XGBoost', xgb_cv.best_estimator_))
# estimator.append(('knn', knn_cv.best_estimator_))
# vot = VotingClassifier(estimators=estimator, voting='hard')
# vot.fit(X_train,y_train)
# print(f'Voting Classifier Train Accuracy Score: {accuracy_score(vot.predict(X_train),y_train)}')
# print(f'Voting Classifier Test Accuracy Score: {accuracy_score(vot.predict(X_test),y_test)}')

# # voting classifier confusion matrix
# vot_predict = vot.predict(X_test)
# vot_test_accuracy = accuracy_score(vot_predict,y_test)
# ConfusionMatrixDisplay(confusion_matrix(vot_predict, y_test)).plot(cmap='Blues')