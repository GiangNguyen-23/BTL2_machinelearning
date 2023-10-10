import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Đọc dữ liệu và chuẩn bị tập huấn luyện
df = pd.read_csv('D:/code hoc may/Breast_cancer_data.csv')
X = df[['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness']].values
y = df['diagnosis'].values

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Xây dựng và huấn luyện mô hình ID3 Decision Tree
id3 = DecisionTreeClassifier(criterion='entropy')
id3.fit(X_train, y_train)

# Dự đoán kết quả trên tập kiểm tra
y_pred = id3.predict(X_test)

# Đánh giá độ chính xác
accuracy = np.mean(y_pred == y_test)
print('Độ chính xác của ID3 Decision Tree:', accuracy)