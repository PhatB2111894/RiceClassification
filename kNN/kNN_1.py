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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import cross_val_score

df = pd.read_excel('C:/Users/tuanp/OneDrive/Documents/GitHub/RiceClassification/Rice_Dataset_Commeo_and_Osmancik/Rice_Dataset_Commeo_and_Osmancik/Rice_Cammeo_Osmancik.xlsx')
df.shape
df.info()
df.describe()
df['Class'].unique()
df['Class'].value_counts()
df.isnull().sum().sort_values(ascending=False)
df.isna().sum().sort_values(ascending=False)
# def handle_iqr_outliers(df, column, remove_outliers=False):
#     # Calculating IQR
#     Q1 = df[column].quantile(0.25)
#     Q3 = df[column].quantile(0.75)
#     IQR = Q3 - Q1

#     # Defining outliers condition
#     outliers_condition = ~((df[column] >= (Q1 - 1.5 * IQR)) & (df[column] <= (Q3 + 1.5 * IQR)))

#     # Counting total outliers
#     total_outliers = outliers_condition.sum()

#     if total_outliers > 0:
#         print(f"Outliers in '{column}': {total_outliers}")
        
#         if remove_outliers:
#             # Removing outliers if specified
#             df.drop(df[outliers_condition].index, inplace=True)
#             print(f"Deleted outliers from '{column}'.")
#     else:
#         print(f"No outliers in '{column}'.")
# for column in df.select_dtypes(include=[np.number]).columns:
#     handle_iqr_outliers(df, column)  # To just count outliers
    
# for column in df.select_dtypes(include=[np.number]).columns:
#     handle_iqr_outliers(df, column, remove_outliers=True)  # To remove outliers
    
var = 'Class'
status_counts = df[var].value_counts()

# plt.figure(figsize=(6, 6))
# status_counts.plot(kind='bar')
# plt.xlabel('Type of rice')
# plt.ylabel('Quantity')
# plt.title('Class distribution')
# plt.xticks(rotation=45)
# plt.show()

# def plot_distribution(df, var, var_title=None):
#     if var_title is None:
#         var_title = var

#     fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(11, 4), sharex=True)

#     sns.boxplot(data=df[var], ax=ax[0], orient='h', width=0.4)
#     ax[0].set_xlabel(var_title)

#     sns.histplot(data=df[var], ax=ax[1], kde=False)
#     ax[1].set_xlabel(var_title)
#     ax[1].set_ylabel('Frequency')

#     sns.kdeplot(data=df[var], ax=ax[2], fill=True)
#     ax[2].set_xlabel(var_title)
#     ax[2].set_ylabel('Density')

#     plt.tight_layout()

# plot_distribution(df, 'Area')
# plot_distribution(df, 'Perimeter')
# plot_distribution(df, 'Major_Axis_Length')
# plot_distribution(df, 'Minor_Axis_Length')
# plot_distribution(df, 'Eccentricity')
# plot_distribution(df, 'Convex_Area')
# plot_distribution(df, 'Extent')

# def plot_class_distribution(df, var, var_title=None):
#     if var_title is None:
#         var_title = var

#     fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), sharex=True)

#     sns.histplot(data=df.loc[df['Class'] == df['Class'].unique()[0]], x=var, label=df['Class'].unique()[0], kde=False, ax=ax[0])
#     sns.histplot(data=df.loc[df['Class'] == df['Class'].unique()[1]], x=var, label=df['Class'].unique()[1], kde=False, ax=ax[0])
#     ax[0].set_xlabel(var_title)
#     ax[0].set_ylabel('Frequency')

#     sns.kdeplot(data=df.loc[df['Class'] == df['Class'].unique()[0]], x=var, label=df['Class'].unique()[0], fill=True, ax=ax[1])
#     sns.kdeplot(data=df.loc[df['Class'] == df['Class'].unique()[1]], x=var, label=df['Class'].unique()[1], fill=True, ax=ax[1])
#     ax[1].set_xlabel(var_title)
#     ax[1].set_ylabel('Density')
#     ax[1].legend()

#     plt.tight_layout()
# plot_class_distribution(df, 'Area')
# plot_class_distribution(df, 'Perimeter')
# plot_class_distribution(df, 'Major_Axis_Length')
# plot_class_distribution(df, 'Minor_Axis_Length')
# plot_class_distribution(df, 'Eccentricity')
# plot_class_distribution(df, 'Convex_Area')
# plot_class_distribution(df, 'Extent')
# df_2 = df.copy()
# sns.pairplot(df_2, hue='Class')


rice_class_0 = df[df['Class'] == 'Cammeo']
rice_class_1 = df[df['Class'] == 'Osmancik']

attributes = ['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length', 'Eccentricity', 'Convex_Area', 'Extent']

for attr in attributes:
    f_statistic, p_value = stats.f_oneway(rice_class_0[attr], rice_class_1[attr])

    # print(f"ANOVA results for '{attr}':")
    # print("F-statistic:", f_statistic)
    # print("p-value:", p_value)

    # if p_value < 0.05:
    #     print("Significant differences exist between at least two of the groups.")
    # else:
    #     print("There is insufficient evidence to support the conclusion that significant differences exist.")
    # print()

# correlation_matrix = df.corr(numeric_only=True)
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
# plt.show()


X = df[['Major_Axis_Length', 'Eccentricity']].values
y = df['Class'].values

le = LabelEncoder()
y = le.fit_transform(y) # [0] Cammeo | [1] Osmancik

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
n = math.ceil(math.sqrt(df.shape[0]))
print(f"N = {n}")
model = KNeighborsClassifier(n_neighbors=n)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# [0] Cammeo | [1] Osmancik
print(classification_report(y_test, y_pred, digits=4))

matrix = confusion_matrix(y_test, y_pred)
display_labels = ['Cammeo', 'Osmancik']

mc_visual = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=display_labels)
mc_visual.plot()
scores = cross_val_score(model, X, y, cv=5)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))