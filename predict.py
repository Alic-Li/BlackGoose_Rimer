import torch
import json
from models.Timer import Model
from utils.data_loader import create_data_loader
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def predict(model, data_loader, scaler):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            input_x, gt = batch
            input_x = input_x.cuda()
            gt = gt.cuda()
            
            # 前向传播
            output = model(input_x, None, None, None, None).cuda()
            output = output[:, :, -1]

            # 反标准化预测结果
            output_denorm = scaler.inverse_transform(output.cpu().numpy())
            gt_denorm = scaler.inverse_transform(gt.cpu().numpy())

            all_preds.extend(output_denorm.flatten())
            all_targets.extend(gt_denorm.flatten())

    return all_preds, all_targets

# 加载配置
config_path = 'config.json'
config = load_config(config_path)

task_name = config['task_name']
ckpt_path = config['ckpt_path']
patch_len = config['patch_len']
d_model = config['d_model']
d_ff = config['d_ff']
e_layers = config['e_layers']
n_heads = config['n_heads']
dropout = config['dropout']
output_attention = config['output_attention']
factor = config['factor']
activation = config['activation']
batch_size = config['batch_size']
num_workers = config['num_workers']
data_path_test = config['data_path_test']
epoch = config['epoch']
save_ckpt_path = config['save_ckpt_path']
window_size = config['window_size']
scaler_path = config['scaler_path']
class Configs:
    def __init__(self):
        self.task_name = task_name
        self.ckpt_path = ckpt_path
        self.patch_len = patch_len
        self.d_model = d_model
        self.d_ff = d_ff
        self.e_layers = e_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.output_attention = output_attention
        self.factor = factor
        self.activation = activation
configs = Configs()
# 加载模型


model = Model(configs).cuda()
model.load_state_dict(torch.load(f'{save_ckpt_path}/best_model.pth'))
model.eval()

# 加载标准化参数
scaler = joblib.load('output_weight/ELC/scaler_weight/scaler.pkl')

# 创建数据加载器
test_data_loader = create_data_loader(data_path_test, window_size, batch_size, num_workers, scaler=scaler)

# 进行预测
all_preds, all_targets = predict(model, test_data_loader, scaler)

# 计算并打印RMSE, MAE, MAPE, R²
rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
mae = mean_absolute_error(all_targets, all_preds)
mape = np.mean(np.abs((np.array(all_targets) - np.array(all_preds)) / np.array(all_targets))) * 100
r2 = r2_score(all_targets, all_preds)

print(f'Prediction - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, R²: {r2:.4f}')