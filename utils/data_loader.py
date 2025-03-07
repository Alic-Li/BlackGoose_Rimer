import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import json

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

class PowerDataset(Dataset):
    def __init__(self, data_path, window_size, target_column=None, scaler=None):
        # 读取 CSV 文件并忽略第一列（时间戳）
        self.data = pd.read_csv(data_path).iloc[:, 1:]
        # 只使用前26000行的数据
        self.data = self.data.iloc[:26000]
        # 将所有列转换为 float32 类型
        self.data = self.data.astype(np.float32)
        config = load_config('config.json')
        scaler_path = config['scaler_path']
        
        if scaler is None:
            # 标准化数据
            self.scaler = StandardScaler()
            self.data_scaled = self.scaler.fit_transform(self.data)
            # 保存标准化参数
            joblib.dump(self.scaler, scaler_path)
        else:
            self.scaler = scaler
            self.data_scaled = self.scaler.transform(self.data)
        
        # 转换为 PyTorch 张量
        self.features = torch.tensor(self.data_scaled, dtype=torch.float32)
        # 设置窗口大小
        self.window_size = window_size
        # 设置目标列，默认为最后一列
        self.target_column = target_column if target_column is not None else self.data.shape[1] - 1

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        # 输入序列: 时间步0到时间步(n-1)
        x = self.features[idx:idx+self.window_size]
        # 目标序列: 时间步1到时间步n
        y = self.features[idx+1:idx+self.window_size+1, self.target_column]
        return x, y

def create_data_loader(data_path, window_size, batch_size, num_workers, target_column=None, scaler=None):
    dataset = PowerDataset(data_path, window_size, target_column=target_column, scaler=scaler)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

if __name__ == '__main__':
    data_loader = create_data_loader('dataset/electricity/train_electricity.csv', window_size=10, batch_size=1, num_workers=4)
    print(len(data_loader))
    for i, (x, y) in enumerate(data_loader):
        print(f"x shape: {x.shape}, y shape: {y.shape}")
        print(f"x mean: {x.mean().item()}, x std: {x.std().item()}")
        print(f"y mean: {y.mean().item()}, y std: {y.std().item()}")
        break