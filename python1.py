import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_iris = pd.read_csv("https://raw.githubusercontent.com/phamdinhkhanh/datasets/master/iris_train.csv", header=0, index_col=None)
print(df_iris.head())
print(df_iris.describe())
# lấy x, y
x = df_iris['Petal.Length'].values
y = df_iris['Petal.Width'].values
# Vẽ biểu đồ line độ dài cánh hoa trung bình giữa các loài hoa
plt.figure(figsize=(16, 8))
plt.scatter(x, y, color='green', )
plt.xlabel('Petal.Length', fontsize=16)
plt.ylabel('Petal.Width', fontsize=16)
plt.title("Average of Petal Length", fontsize=18)
plt.show()
