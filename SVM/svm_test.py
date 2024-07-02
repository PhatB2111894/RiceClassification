import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
# Load the dataset
df = pd.read_excel('C:/Users/tuanp/OneDrive/Documents/GitHub/RiceClassification/Rice_Dataset_Commeo_and_Osmancik/Rice_Dataset_Commeo_and_Osmancik/Rice_Cammeo_Osmancik.xlsx')

# Select features and target variable
X = df.drop('Class', axis=1).values
y = df['Class'].values

# Encode target variable
le = LabelEncoder()
y = le.fit_transform(y)  # [0] Cammeo | [1] Osmancik

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('classifier', SVC(C=10000, kernel='linear', gamma=0.01))
])

# Fit the model using pipeline
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Evaluate model performance
print(classification_report(y_test, y_pred, digits=4))

# Calculate and print accuracy using cross-validation
scores = cross_val_score(pipeline, X, y, cv=5)
print(scores)
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

# Calculate and print precision, recall, and F1-score using cross-validation
scores_precision = cross_val_score(pipeline, X, y, cv=5, scoring='precision')
scores_recall = cross_val_score(pipeline, X, y, cv=5, scoring='recall')
scores_f1 = cross_val_score(pipeline, X, y, cv=5, scoring='f1')

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