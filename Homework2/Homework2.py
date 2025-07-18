import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
data = load_breast_cancer()
x = data.data  # 提取每个样本的特征数据 (569, 30)
y = data.target  # 提取每个样本对应的标签 (569)
# print(data.keys())

# 2. 标准化，使每一列均值为0，方差为1
scaler = StandardScaler()
x_std = scaler.fit_transform(x)

# 3. 记录不同维度下的准确率
dimension = list(range(1, 31))  # PCA降维到 1 ～ 30
accuracies = []

for d in dimension:
    # 将标准化后的数据降维到 d 维
    pca = PCA(n_components=d)
    x_pca = pca.fit_transform(x_std)

    # 划分训练集和测试集，30% 作为测试集，70% 作为训练集
    x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.3, random_state=0)

    # 设置最大迭代次数为1000，使用划分出的训练集来训练线形回归模型
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    # 预测并计算准确率
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# 4. 使用未降维的数据训练线形回归作为对照
x_train_full, x_test_full, y_train_full, y_test_full = train_test_split(x, y, test_size=0.3, random_state=0)
model_full = LogisticRegression(max_iter=1000)
model_full.fit(x_train_full, y_train_full)
y_pred_full = model_full.predict(x_test_full)
accuracy_full1 = accuracy_score(y_test_full, y_pred_full)

# 5. 使用未降维但标准化的数据训练线形回归
x_train_full, x_test_full, y_train_full, y_test_full = train_test_split(x_std, y, test_size=0.3, random_state=0)
model_full = LogisticRegression(max_iter=1000)
model_full.fit(x_train_full, y_train_full)
y_pred_full = model_full.predict(x_test_full)
accuracy_full2 = accuracy_score(y_test_full, y_pred_full)

# 5. 可视化 pca 不同降维维度对于正确率的影响
plt.figure(figsize=(12, 6))
plt.plot(dimension, accuracies, marker='o', label='PCA降维后模型')
plt.hlines(accuracy_full1, xmin=1, xmax=30, colors='r', linestyles='--', label='原始数据（未标准化）')
plt.hlines(accuracy_full2, xmin=1, xmax=30, colors='g', linestyles='--', label='原始数据（标准化）')
plt.title('不同PCA维度与模型准确率对比')
plt.xlabel('PCA降维维度数')
plt.ylabel('测试集准确率')
plt.xticks(dimension)
plt.ylim(0.8, 1.02)
plt.grid(True)
plt.legend()
plt.show()