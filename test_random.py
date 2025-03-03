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

# if __name__ == '__main__':
#     import torch
#     from time_series_library.models import TimeMixer as TM

#     class Configs:
#         def __init__(self):
#             self.task_name = 'long_term_forecast'
#             self.seq_len = 96
#             self.label_len = 48
#             self.pred_len = 96
#             self.down_sampling_window = 2
#             self.down_sampling_layers = 3
#             self.channel_independence = 0
#             self.d_model = 512
#             self.d_ff = 2048
#             self.embed = 'fixed'
#             self.freq = 'h'
#             self.dropout = 0.1
#             self.use_norm = 1
#             self.moving_avg = 25
#             self.decomp_method = 'dft_decomp'
#             self.top_k = 5
#             self.enc_in = 7
#             self.c_out = 7
#             self.num_class = 10
#             self.e_layers = 2
#             self.down_sampling_method = 'avg'

#     configs = Configs()
#     print(configs)

#     # 初始化模型
#     model = TM.TimeMixer(configs).cuda()

#     # 创建随机张量作为输入
#     B, L, M = 1, configs.seq_len, configs.enc_in  # 批量大小、序列长度、特征数量
#     x_enc = torch.randn(B, L, M).cuda()


#     # 前向传播
#     output = model(x_enc, None, None, None, None).cuda()

#     print("Output shape:", output.shape)
#     print(output)  # 输出形状应为 [B, pred_len, c_out]