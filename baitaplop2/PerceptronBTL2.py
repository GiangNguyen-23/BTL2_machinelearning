import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron

df = pd.read_csv('D:/code hoc may/Breast_cancer_data.csv') #đọc tập csv và lưu trữ dạng dataframe
X_data = np.array(df[['mean_radius','mean_texture','mean_perimeter','mean_area','mean_smoothness','diagnosis']].values) # lấy giá trị của các cột được chọn từ df và chuyển đổi thành mảng numpy
data=X_data  #gán cho data
dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle = True)
# print(dt_Train)

X_train = dt_Train[:, :5]
y_train = dt_Train[:, 5]
X_test = dt_Test[:, :5]
y_test = dt_Test[:, 5]
#clf = Perceptron(penalty=None, alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, eta0=1.0, n_jobs=None, random_state=0, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False)
clf = Perceptron(alpha=0.001, max_iter=10000, random_state=0,tol=0.001)
clf.fit(X_train, y_train)

# Dự đoán kết quả trên tập kiểm tra
y_pred = clf.predict(X_test)

# Đánh giá độ chính xác
accuracy = np.mean(y_pred == y_test)
print('Độ chính xác của Perceptron:', accuracy)