# -*- coding:utf-8 -*-
'''
model inference of PR(Polynomial Regression)
'''
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
import pandas as pd
import pickle
import datetime

# 加载测试数据
test_data = pd.read_csv('test.csv', header=None, skiprows=1, encoding='ISO-8859-1')

# 对象类型列进行标签编码
object_columns = test_data.select_dtypes(include=['object']).columns
for col in object_columns:
    le = LabelEncoder()
    test_data[col] = le.fit_transform(test_data[col])

# 提取特征
X_test = test_data.iloc[:, 1:-1]
y_test = test_data.iloc[:, -1]

# 生成多项式特征
poly = PolynomialFeatures(degree=1)  # 与训练时保持一致
X_test_poly = poly.fit_transform(X_test)

# 标准化
scaler = StandardScaler()__
X_test_scaled = scaler.fit_transform(X_test_poly)

# 预测
with open('PRmodel.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

predictions = loaded_model.predict(X_test_scaled)

# 存储结果
all_results = []
for i, index in enumerate(X_test_scaled):
    all_results.append({
        'Index': index,
        'Features': X_test_scaled[i].tolist(),
        'Predicted': predictions[i],
        'Actual': y_test.iloc[i]
    })
    
print(predictions)
prediction_df = pd.DataFrame(all_results)
# 获取当前时间
current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
filename = f'PR_prediction_results_{current_time}.csv'
prediction_df.to_csv(filename, index=False)
