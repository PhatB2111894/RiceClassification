import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_excel('C:/Users/tuanp/OneDrive/Documents/GitHub/RiceClassification/Rice_Dataset_Commeo_and_Osmancik/Rice_Dataset_Commeo_and_Osmancik/Rice_Cammeo_Osmancik.xlsx')

# Inspect the column names
print("Column names in the dataset:")
print(df.columns)

# Select features and target variable
X = df.drop('Class', axis=1)
y = df['Class']

# Encode target variable
le = LabelEncoder()
y = le.fit_transform(y)  # [0] Cammeo | [1] Osmancik

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create DataFrame for scaled data
df_after_scaling = pd.DataFrame(X_train_scaled, columns=X.columns)

# Calculate min, mean, max, and standard deviation for each feature after scaling
stats_after_scaling = df_after_scaling.describe().loc[['min', 'mean', 'max', 'std']]

# Rename columns for the stats DataFrame
stats_after_scaling.columns = [f'After Scaling {col}' for col in stats_after_scaling.columns]

# Combine stats into one DataFrame for display
stats_combined = pd.concat([stats_after_scaling], axis=1)
print(stats_combined)

# Adjust feature list based on actual column names
features = [col for col in ['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length',
                            'Eccentricity', 'Convex_Area', 'Extent'] if col in X.columns]

# Plot histograms before scaling
plt.figure(figsize=(20, 15))
plt.suptitle('Histograms of Features Before Scaling', fontsize=16)
for i, feature in enumerate(features):
    plt.subplot(3, 3, i + 1)
    plt.hist(X[feature], bins=30, alpha=0.75)
    plt.title('')  # Remove default title
    plt.xlabel(f'({chr(97 + i)}) {feature}')  # Feature name as xlabel with alphabet
    plt.ylabel('Frequency')  # ylabel for histogram
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the suptitle
plt.show()

# Plot histograms after scaling
plt.figure(figsize=(20, 15))
plt.suptitle('Histograms of Features After Scaling', fontsize=16)
for i, feature in enumerate(features):
    plt.subplot(3, 3, i + 1)
    plt.hist(df_after_scaling[feature], bins=30, alpha=0.75)
    plt.title('')  # Remove default title
    plt.xlabel(f'({chr(97 + i)}) {feature}')  # Feature name as xlabel with alphabet
    plt.ylabel('Frequency')  # ylabel for histogram
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the suptitle
plt.show()
