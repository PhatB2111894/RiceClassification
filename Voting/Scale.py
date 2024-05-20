import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_excel('C:/Users/tuanp/OneDrive/Documents/GitHub/RiceClassification/Rice_Dataset_Commeo_and_Osmancik/Rice_Dataset_Commeo_and_Osmancik/Rice_Cammeo_Osmancik.xlsx')

# Select features and target variable
X = df.drop('Class', axis=1).values
y = df['Class'].values

# Encode target variable
le = LabelEncoder()
y = le.fit_transform(y)  # [0] Cammeo | [1] Osmancik

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Tạo DataFrame chứa dữ liệu trước và sau khi chuẩn hóa cho tất cả các cột thuộc tính

df_after_scaling = pd.DataFrame(X_train_scaled, columns=df.drop('Class', axis=1).columns)

# Tính min, mean, max và standard deviation cho từng đặc trưng

stats_after_scaling = df_after_scaling.describe().loc[['min', 'mean', 'max', 'std']]

# Đặt tên cột cho dữ liệu trước và sau khi chuẩn hóa

stats_after_scaling.columns = [f'After Scaling {col}' for col in stats_after_scaling.columns]

# Kết hợp các DataFrame để hiển thị
stats_combined = pd.concat([stats_after_scaling], axis=1)
print(stats_combined)

# Tên file Excel đích
excel_file_path = 'stats_before_and_after_scaling.xlsx'

# Ghi dữ liệu vào file Excel
with pd.ExcelWriter(excel_file_path) as writer:
    # Ghi dữ liệu trước và sau khi chuẩn hóa vào các sheet riêng biệt
    stats_after_scaling.to_excel(writer, sheet_name='After Scaling')
    print("Excel file saved successfully!")
