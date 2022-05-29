from tcn_class import ResidualBlock
from torch import nn

class TCN_autoencoder(nn.Module):
    def __init__(self, chan_list, kernel_size, dropout=0.2):
        super(TCN_autoencoder, self).__init__()    

        layers = []
        dilation = 1
        for i in range(len(chan_list)//2):
            layers.append(ResidualBlock(chan_list[i], chan_list[i+1], kernel_size, dilation, dropout))
            layers.append(nn.AvgPool1d(2,2))
        for j in range(len(chan_list)//2):
            layers.append(ResidualBlock(chan_list[j+i+1], chan_list[j+i+2], kernel_size, dilation, dropout))
            layers.append(nn.Upsample(scale_factor=2, mode='linear', align_corners=True))

        layers.append(ResidualBlock(chan_list[0], chan_list[-1], kernel_size, dilation, dropout))
        layers.append(ResidualBlock(chan_list[0], chan_list[-1], kernel_size, dilation, dropout))
        layers.append(ResidualBlock(chan_list[0], chan_list[-1], kernel_size, dilation, dropout))

        self.model = nn.Sequential(*layers)
   
    def forward(self, X):
         
        return self.model(X)


# class TCN_autoencoder(nn.Module):
#     def __init__(self, scale_factor, in_channels, kernel_size, dropout=0.2):
#         super(TCN_autoencoder, self).__init__()

#         layers = []
#         dilation = 1
#         cur_ch = in_channels
#         for _ in range(scale_factor):
#             layers.append(ResidualBlock(cur_ch, cur_ch*2, kernel_size, dilation, dropout))
#             layers.append(nn.AvgPool1d(2,2))
#             cur_ch *= 2
#         for _ in range(scale_factor):
#             layers.append(ResidualBlock(cur_ch, cur_ch//2, kernel_size, dilation, dropout))
#             layers.append(nn.Upsample(scale_factor=2, mode='linear', align_corners=True))
#             cur_ch //= 2

#         layers.append(ResidualBlock(in_channels, in_channels, kernel_size, dilation, dropout))

#         self.model = nn.Sequential(*layers)
   
#     def forward(self, X):
         
#         return self.model(X)


# import pickle
# import numpy as np
# from torch import nn
# import torch
# import matplotlib.pyplot as plt
# from tqdm.notebook import tqdm

# WIN_SIZE = 64
# STRIDE = 4
# PATH = 'dil_sim_test'
# with open('data/{}/train.pkl'.format(PATH), 'rb') as f:
#     train_df = pickle.load(f)

# def prep_df(df, in_channels, win_size, stride):
#     series_len = len(df[0])
#     df_size = ((series_len - win_size) // stride) + 1
#     torch_df = torch.FloatTensor(df_size, in_channels, win_size)
#     for i in range(df_size):
#         sample = df[:, stride*i: stride*i+win_size]
#         torch_df[i] = sample
#     residual_series = df[:, -win_size:].reshape(1,in_channels, -1)
#     # torch_df = torch.clamp(torch_df, 0, 1)
#     # residual_series = torch.clamp(residual_series, 0, 1)
#     return torch_df, residual_series

# in_channels = len(train_df)
# train, train_resid = prep_df(train_df, in_channels, WIN_SIZE, STRIDE) 
# train_size = train.shape[0]

# ks = 5
# sf = int(np.log2((WIN_SIZE+ks)/ks))

# ch_list = [in_channels, 40, 60, 80, 60, 40, in_channels]
# model = TCN_autoencoder(ch_list, kernel_size=ks, dropout=0) 
# # model.load_state_dict(torch.load('saved_models/shift_sim_64_4.pt'))
# # model.eval()

# loss_history = []

# epochs_n = 1000
# learning_rate = 1e-3
# criterion = nn.MSELoss()
# optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

# for ep in tqdm(range(epochs_n)):
#     ep_loss = 0
#     pred = model(train)
#     loss = criterion(pred, train)

#     pred_resid = model(train_resid)
#     loss_resid = criterion(pred_resid, train_resid)
#     loss = (loss*train_size + loss_resid) / (train_size + 1)

#     optim.zero_grad()
#     loss.backward()
#     optim.step()

#     loss_history.append(loss.item())

#     if ep % 5 == 0:
#         print(loss.item()) 

# print(loss.item()) 
# plt.figure(figsize=(12,6))
# plt.plot(loss_history[:]) 