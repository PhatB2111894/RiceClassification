import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
# Load the dataset
df = pd.read_excel('C:/Users/tuanp/OneDrive/Documents/GitHub/RiceClassification/Rice_Dataset_Commeo_and_Osmancik/Rice_Dataset_Commeo_and_Osmancik/Rice_Cammeo_Osmancik.xlsx')

X = df.drop('Class', axis=1).values
y = df['Class'].values

# Encode target variable
le = LabelEncoder()
y = le.fit_transform(y)  # [0] Cammeo | [1] Osmancik

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

grid = {
    'n_estimators': [300],
    'criterion': ['gini'],
    'max_depth': [10],
    'min_samples_split': [5],
    'min_samples_leaf': [1],
    'bootstrap': [True]
}
# Initialize RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Perform grid search with cross-validation
rf_cv = GridSearchCV(rf, grid, cv=kf,verbose=0, scoring='accuracy', refit=True)
rf_cv.fit(X_train, y_train)

# Predict on the test set using the GridSearchCV object
y_pred_rf = rf_cv.predict(X_test)
# Print the best parameters and estimator
print(f'Best parameters found: {rf_cv.best_params_}')
print(f'Best estimator: {rf_cv.best_estimator_}')
# Calculate and print accuracy using cross-validation
scores = cross_val_score(rf_cv, X, y, cv=5)
print(scores)
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

# Calculate and print precision, recall, and F1-score using cross-validation
scorers = {
    'precision': make_scorer(precision_score, average=None),
    'recall': make_scorer(recall_score, average=None),
    'f1_score': make_scorer(f1_score, average=None)
}

scores_precision = cross_val_score(rf_cv, X, y, cv=5, scoring='precision')
scores_recall = cross_val_score(rf_cv, X, y, cv=5, scoring='recall')
scores_f1 = cross_val_score(rf_cv, X, y, cv=5, scoring='f1')

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
conf_matrix = confusion_matrix(y_test, y_pred_rf, normalize='true')

# Save normalized confusion matrix to a DataFrame
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Cammeo', 'Osmancik'], columns=['Cammeo', 'Osmancik'])
print(conf_matrix_df)