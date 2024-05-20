from scipy.io import arff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
# Load the dataset
data = arff.loadarff('C:/Users/tuanp/OneDrive/Documents/GitHub/RiceClassification/Rice_Dataset_Commeo_and_Osmancik/Rice_Dataset_Commeo_and_Osmancik/Rice_Cammeo_Osmancik.arff')
df = pd.DataFrame(data[0])
df['Class'] = df['Class'].str.decode('utf-8')

# Check the distribution of the target variable
print(df['Class'].value_counts(normalize=True))

# Split the dataset into features and target variable, and standardize features
X = df.iloc[:,:-1].to_numpy()
Y = df.iloc[:,-1].astype('category').cat.codes.to_numpy()
scaler = StandardScaler()
standard_x = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(standard_x, Y, test_size=0.2, random_state=42, stratify=Y)

# Define KFold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Define XGBoost parameters for grid search
xgb_params = {'n_estimators': [100],
              'max_depth': [3],
              'gamma': [0.4],
              'reg_alpha': [0.01]}

# Initialize XGBoost Classifier
xgb = XGBClassifier(learning_rate=0.1, random_state=42)

# Grid search for hyperparameter tuning
xgb_cv = GridSearchCV(xgb, xgb_params, cv=kf, refit=True, scoring='accuracy', verbose=0)
xgb_cv.fit(X_train, y_train)

y_pred_test = xgb_cv.predict(X_test)
# Calculate and print accuracy using cross-validation
scores = cross_val_score(xgb_cv, X, Y, cv=5)
print(scores)
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

# Calculate and print precision, recall, and F1-score using cross-validation
scorers = {
    'precision': make_scorer(precision_score, average=None),
    'recall': make_scorer(recall_score, average=None),
    'f1_score': make_scorer(f1_score, average=None)
}

scores_precision = cross_val_score(xgb_cv, X, Y, cv=5, scoring='precision')
scores_recall = cross_val_score(xgb_cv, X, Y, cv=5, scoring='recall')
scores_f1 = cross_val_score(xgb_cv, X, Y, cv=5, scoring='f1')

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
conf_matrix = confusion_matrix(y_test, y_pred_test, normalize='true')

# Save normalized confusion matrix to a DataFrame
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Cammeo', 'Osmancik'], columns=['Cammeo', 'Osmancik'])
print(conf_matrix_df)
