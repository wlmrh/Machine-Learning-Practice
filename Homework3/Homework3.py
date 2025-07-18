import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam

# 1. 加载数据
housing = fetch_california_housing()
X, y = housing.data, housing.target
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 数据标准化
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 3. 定义传统模型
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'SVM': SVR(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = (mse, r2)

# 4. 基线神经网络模型
def build_baseline_nn():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(x_train_scaled.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.01),
                  loss='mse',
                  metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return model

# 5. BatchNorm 神经网络模型
def build_batchnorm_nn():
    model = Sequential([
        Dense(64, use_bias=False, input_shape=(x_train_scaled.shape[1],)),
        BatchNormalization(),
        Activation('relu'),
        Dense(32, use_bias=False),
        BatchNormalization(),
        Activation('relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.01),
                  loss='mse',
                  metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return model

baseline_model = build_baseline_nn()
batchnorm_model = build_batchnorm_nn()

history_base = baseline_model.fit(x_train_scaled, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=0)
history_bn = batchnorm_model.fit(x_train_scaled, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=0)

# 6. 可视化 RMSE 曲线
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history_base.history['rmse'], label='Baseline Train')
plt.plot(history_base.history['val_rmse'], label='Baseline Val')
plt.plot(history_bn.history['rmse'], label='BatchNorm Train')
plt.plot(history_bn.history['val_rmse'], label='BatchNorm Val')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('RMSE Comparison')
plt.legend()

# 7. 可视化 Loss 曲线
plt.subplot(1,2,2)
plt.plot(history_base.history['loss'], label='Baseline Train')
plt.plot(history_base.history['val_loss'], label='Baseline Val')
plt.plot(history_bn.history['loss'], label='BatchNorm Train')
plt.plot(history_bn.history['val_loss'], label='BatchNorm Val')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Loss Comparison')
plt.legend()

plt.tight_layout()
plt.show()

# 8. 评估神经网络模型
mse_base, rmse_base = baseline_model.evaluate(x_test_scaled, y_test, verbose=0)
mse_bn, rmse_bn = batchnorm_model.evaluate(x_test_scaled, y_test, verbose=0)

# 9. 加入结果表格
results['Baseline NN'] = (mse_base, r2_score(y_test, baseline_model.predict(x_test_scaled)))
results['BatchNorm NN'] = (mse_bn, r2_score(y_test, batchnorm_model.predict(x_test_scaled)))

# 10. 输出所有模型结果
df_results = pd.DataFrame.from_dict(
    {k: {'MSE': v[0], 'R2': v[1]} for k, v in results.items()},
    orient='index'
)
print(df_results)
df_results
