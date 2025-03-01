import torch
import json
from models.Timer import Model
from utils.data_loader import create_data_loader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def train(config):
    # 加载配置
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
    data_path = config['data_path']
    epoch = config['epoch']
    save_ckpt_path = config['save_ckpt_path']
    window_size = config['window_size']

    # 初始化配置类
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

    # 初始化模型
    model = Model(configs).cuda()

    # 创建数据加载器
    data_loader = create_data_loader(data_path, window_size, batch_size, num_workers)
    print('Data loaded! Start training. Ciallo~(∠・ω< )⌒☆')

    # 初始化优化器，降低学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001) 

    # 初始化调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # 训练循环
    for epoch in range(epoch):
        model.train()
        total_loss = 0
        all_preds = []
        all_targets = []

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 创建 tqdm 对象，添加学习率信息
        data_loader_tqdm = tqdm(data_loader, desc=f'Epoch {epoch+1} (LR: {current_lr:.6f})', postfix=dict(loss=0.0))

        for batch in data_loader_tqdm:

            input_x, gt = batch
            input_x = input_x.cuda()
            gt = gt.cuda()
            
            # 前向传播
            output = model(input_x, None, None, None, None).cuda()
            output = output[:, :, -1]

            # 计算损失
            loss = torch.nn.MSELoss()(output, gt) 
            data_loader_tqdm.set_postfix(loss=loss.item())

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # 调整梯度裁剪的最大范数
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend(output.cpu().detach().numpy())
            all_targets.extend(gt.cpu().detach().numpy())
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        avg_loss = total_loss / len(data_loader)
        print("=============================================================================")
        print(f'Epoch [{epoch+1}/{epoch}], Loss: {avg_loss:.4f}')

        # 计算并打印RMSE, MAE, MAPE, R²
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        mae = mean_absolute_error(all_targets, all_preds)
        mape = np.mean(np.abs((np.array(all_targets) - np.array(all_preds)) / np.array(all_targets))) * 100
        r2 = r2_score(all_targets, all_preds)

        print(f'Epoch [{epoch+1}/{epoch}] - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, R²: {r2:.4f}')
        print("=============================================================================")

        # 更新调度器
        scheduler.step(avg_loss)

        # 保存模型
        torch.save(model.state_dict(), f'{save_ckpt_path}/model.pth')

if __name__ == '__main__':
    config_path = 'config.json'
    config = load_config(config_path)
    train(config)