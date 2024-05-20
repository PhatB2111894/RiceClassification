import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
# Load the dataset
df = pd.read_excel('C:/Users/tuanp/OneDrive/Documents/GitHub/RiceClassification/Rice_Dataset_Commeo_and_Osmancik/Rice_Dataset_Commeo_and_Osmancik/Rice_Cammeo_Osmancik.xlsx')

X = df.drop('Class', axis=1).values
y = df['Class'].values

# Encode target variable
le = LabelEncoder()
y = le.fit_transform(y)  # [0] Cammeo | [1] Osmancik

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

grid = {'n_estimators':[200],
        'criterion':['entropy']
       }
rf = RandomForestClassifier(random_state = 123)
rf_cv = GridSearchCV(rf,grid,cv=5)
rf_cv.fit(X_train,y_train)

# Predict on the test set using the GridSearchCV object
y_pred_rf = rf_cv.predict(X_test)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, digits=4))

# Generate confusion matrix visualization
matrix = confusion_matrix(y_test, y_pred_rf)
display_labels = ['Cammeo', 'Osmancik']
mc_visual = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=display_labels)
mc_visual.plot()