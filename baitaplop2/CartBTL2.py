import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Đọc dữ liệu và chuẩn bị tập huấn luyện
df = pd.read_csv('D:/code hoc may/Breast_cancer_data.csv') #đọc tập csv và lưu trữ dạng dataframe
X_data = np.array(df[['mean_radius','mean_texture','mean_perimeter','mean_area','mean_smoothness','diagnosis']].values) # lấy giá trị của các cột được chọn từ df và chuyển đổi thành mảng numpy
data=X_data  #gán cho data
dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle = True)
dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle = True)
np.set_printoptions(suppress=True)
print(dt_Train)
X_train = dt_Train[:, :5]
y_train = dt_Train[:, 5]
X_test = dt_Test[:, :5]
y_test = dt_Test[:, 5]

# Xây dựng và huấn luyện mô hình Cart Decision Tree
clf = DecisionTreeClassifier(criterion='gini', splitter='best',  min_samples_split=2)

clf.fit(X_train, y_train)

# Dự đoán kết quả trên tập kiểm tra
y_pred = clf.predict(X_test)

# Đánh giá độ chính xác
accuracy = np.mean(y_pred == y_test)
print('Độ chính xác của CART Decision Tree:', accuracy)