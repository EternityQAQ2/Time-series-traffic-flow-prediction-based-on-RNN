from matplotlib import pyplot as plt
import mindspore.context as context
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import mindspore.nn as nn
from mindspore.train.callback import LossMonitor
from mindspore.dataset import NumpySlicesDataset
import mindspore
from mindspore import Tensor
from mindspore.train import Model
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, filename='training_log.log', filemode='w', 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 数据加载
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Dataset loaded from {file_path}.")
        return data
    except Exception as e:
        logging.error(f"Error loading dataset from {file_path}: {e}")
        return None

# 加载训练和测试数据
def load_all_data():
    X_train = load_data('./saved_data/X_train.csv')
    y_train = load_data('./saved_data/y_train.csv')
    X_test = load_data('./saved_data/X_test.csv')
    y_test = load_data('./saved_data/y_test.csv')
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_all_data()

# 查看数据的前几行，检查时间戳列
logging.info("Previewing training data:")
logging.info(X_train.head())

# 假设时间戳是数据集的第一列，将其转换为datetime类型
X_train['timestamp'] = pd.to_datetime(X_train.iloc[:, 0])
X_test['timestamp'] = pd.to_datetime(X_test.iloc[:, 0])

# 提取时间特征
X_train['year'] = X_train['timestamp'].dt.year
X_train['month'] = X_train['timestamp'].dt.month
X_train['day'] = X_train['timestamp'].dt.day
X_train['hour'] = X_train['timestamp'].dt.hour
X_test['year'] = X_test['timestamp'].dt.year
X_test['month'] = X_test['timestamp'].dt.month
X_test['day'] = X_test['timestamp'].dt.day
X_test['hour'] = X_test['timestamp'].dt.hour

# 删除原始的时间戳列
X_train.drop(columns=['timestamp'], inplace=True)
X_test.drop(columns=['timestamp'], inplace=True)

# 确保 X_train 和 X_test 只有数值列
X_train = X_train.select_dtypes(include=[np.number])
X_test = X_test.select_dtypes(include=[np.number])

logging.info(f"X_train columns after processing: {X_train.columns}")

# 进行数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape for GRU input: (batch_size, seq_len, feature_size)
X_train_scaled = X_train_scaled[:, np.newaxis, :]
X_test_scaled = X_test_scaled[:, np.newaxis, :]

logging.info(f"X_train_scaled shape: {X_train_scaled.shape}, X_test_scaled shape: {X_test_scaled.shape}")

# 构建GRU模型
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

class GRUNetwork(nn.Cell):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout_rate=0.2):
        super(GRUNetwork, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.fc = nn.SequentialCell([
            nn.Dense(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dense(hidden_size // 2, output_size)
        ])
    
    def construct(self, x):
        gru_out, _ = self.gru(x)  # GRU 输出: (batch_size, seq_len, hidden_size)
        output = self.fc(gru_out[:, -1, :])  # 取最后一个时间步 (batch_size, hidden_size)
        return output

# 确保所有数据的类型一致
X_train_scaled = X_train_scaled.astype(np.float32)
X_test_scaled = X_test_scaled.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# 模型参数设置
input_size = X_train_scaled.shape[2]  # 输入特征维度
hidden_size = 128  # 隐藏层单元数
output_size = 1  # 输出维度
num_layers = 3  # GRU 层数

# 初始化模型
model = GRUNetwork(input_size, hidden_size, output_size, num_layers=num_layers, dropout_rate=0.3)

# 定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = nn.Adam(model.trainable_params(), learning_rate=0.001)

# 创建训练模型
train_model = Model(model, loss_fn, optimizer)

# 构造训练数据集
train_dataset = NumpySlicesDataset((X_train_scaled, y_train.values), shuffle=True)
train_dataset = train_dataset.batch(32)  # 设置 batch 大小

# 训练模型
epochs = 50  # 设置训练的总轮数
best_mae = float('inf')  # 用于记录最佳 MAE

logging.info("Starting training...")
train_model.train(epochs, train_dataset, callbacks=[LossMonitor()])
logging.info("Training complete.")

# 保存模型
checkpoint_path = './saved_model/gru_model.ckpt'  # 设置保存路径
mindspore.save_checkpoint(model, checkpoint_path)
logging.info(f"Model saved to {checkpoint_path}")

# 每轮训练后进行评估
X_test_tensor = Tensor(X_test_scaled, dtype=mindspore.float32)
y_pred = train_model.predict(X_test_tensor)
y_pred = y_pred.asnumpy().flatten()  # 转为 NumPy 并展开为 1D
mae = mean_absolute_error(y_test.values, y_pred)
logging.info(f"Final Validation MAE: {mae:.4f}")

# 如果当前模型在验证集上的MAE更小，则更新最佳模型
if mae < best_mae:
    best_mae = mae
    logging.info("Best model updated.")

# 计算 RMSE
rmse = np.sqrt(mean_squared_error(y_test.values, y_pred))
logging.info(f"Final RMSE: {rmse:.4f}")

# 结果可视化    
def plot_results(y_true, y_pred, title="Prediction vs True Values"):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label="True Values", alpha=0.7)
    plt.plot(y_pred, label="Predicted Values", alpha=0.7)
    plt.legend()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Total Amount")
    # 保存图像
    image_path = './predictions_vs_true_values.png'  # 设置保存图像的路径
    plt.savefig(image_path)
    logging.info(f"Plot saved to {image_path}")
    plt.show()

# 可视化结果
plot_results(y_test.values, y_pred)