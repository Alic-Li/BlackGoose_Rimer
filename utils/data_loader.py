import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class PowerDataset(Dataset):
    def __init__(self, data_path, sequence_length, target_column=None):
        # 读取 CSV 文件并忽略第一列（时间戳）
        self.data = pd.read_csv(data_path).iloc[:, 1:]
        # 只使用前26000行的数据
        self.data = self.data.iloc[:26000]
        # 将所有列转换为 float32 类型
        self.data = self.data.astype(np.float32)
        # 转换为 PyTorch 张量
        self.features = torch.tensor(self.data.values, dtype=torch.float32)
        # 设置序列长度
        self.sequence_length = sequence_length
        # 设置目标列，默认为最后一列
        self.target_column = target_column if target_column is not None else self.data.shape[1] - 1

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        x = self.features[idx:idx+self.sequence_length]
        return x

def create_data_loader(data_path, sequence_length, batch_size, num_workers, target_column=None):
    dataset = PowerDataset(data_path, sequence_length, target_column=target_column)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

def remove_last_seven(tensor):

    if tensor.dim() != 3:
        raise ValueError("输入张量必须是三维张量")
    
    batch_size, sequence_length, features = tensor.shape
    if sequence_length <= 7:
        raise ValueError("序列长度必须大于7")
    
    return tensor[:, :-7, :]

def extract_last_seven(tensor):

    if tensor.dim() != 3:
        raise ValueError("输入张量必须是三维张量")
    
    batch_size, sequence_length, features = tensor.shape
    if sequence_length < 7:
        raise ValueError("序列长度必须至少为7")
    
    return tensor[:, -7:, :]

if __name__ == '__main__':
    data_loader = create_data_loader('/home/alic-li/PowerForecasting/dataset/electricity/electricity.csv', sequence_length=1000, batch_size=1, num_workers=4)
    print(len(data_loader))
    for i, batch in enumerate(data_loader):
        x = batch
        print(f"x shape: {x.shape}")  # 输出形状应为 [batch_size, sequence_length, features]
        break