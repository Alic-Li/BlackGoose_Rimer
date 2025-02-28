if __name__ == '__main__':
    import torch
    from models.Timer import Model
    class Configs:
        def __init__(self):
            self.task_name = 'forecast' 
            self.ckpt_path = ''  # 如果有预训练模型路径，可以在这里指定
            self.patch_len = 16
            self.d_model = 512
            self.d_ff = 2048
            self.e_layers = 6
            self.n_heads = 8
            self.dropout = 0.1
            self.output_attention = False
            self.factor = 3
            self.activation = 'gelu'


    configs = Configs()
    print(configs)

    # 初始化模型
    model = Model(configs).cuda()

    # 创建随机张量作为输入
    B, L, M = 1000, 96, 7  # 批量大小、序列长度、特征数量
    x_enc = torch.randn(B, L, M).cuda()


    # 前向传播
    output = model(x_enc, None, None, None, None).cuda()

    print(output.shape)
    print(output)  # 输出形状应为 [B, T, D]