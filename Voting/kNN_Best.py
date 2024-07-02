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
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
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
X_train, X_test, y_train, y_test = train_test_split(standard_x, Y, test_size=1/3, random_state=42, stratify=Y)
# print(f'train feature shape: {X_train.shape}')
# print(f'train target shape: {y_train.shape}')
# print(f'test feature shape: {X_test.shape}')
# print(f'test feature shape: {y_test.shape}')

# establish kfold
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# knn with parameter tuning on grid search
knn_params = {
    'n_neighbors': [120],
    'weights': ['distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': [1]
}
knn_model = KNeighborsClassifier()
knn_cv = GridSearchCV(knn_model, knn_params, cv=kf, verbose=0, scoring='accuracy', refit=True)
knn_cv.fit(X_train, y_train)
y_pred = knn_cv.predict(X_test)
print(f'Training Accuracy of KNN: {knn_cv.best_score_}')
print(f'Estimator of KNN: {knn_cv.best_estimator_}')
print(classification_report(y_test, y_pred, digits=4))
scores = cross_val_score(knn_cv, X, Y, cv=5)
print(scores)
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

# Calculate and print precision, recall, and F1-score using cross-validation
scorers = {
    'precision': make_scorer(precision_score, average=None),
    'recall': make_scorer(recall_score, average=None),
    'f1_score': make_scorer(f1_score, average=None)
}

# Use original target array Y for cross-validation
scores_precision = cross_val_score(knn_cv, X, Y, cv=5, scoring='precision')
scores_recall = cross_val_score(knn_cv, X, Y, cv=5, scoring='recall')
scores_f1 = cross_val_score(knn_cv, X, Y, cv=5, scoring='f1')

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
conf_matrix = confusion_matrix(y_test, y_pred, normalize='true')

# Save normalized confusion matrix to a DataFrame
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Cammeo', 'Osmancik'], columns=['Cammeo', 'Osmancik'])
print(conf_matrix_df)