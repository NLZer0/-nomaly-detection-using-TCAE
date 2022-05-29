import torch
from torch import nn
from torch import serialization
from torch.nn.utils import weight_norm

# A class describing blocks with a residual connection
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(ResidualBlock, self).__init__()
        
        self.kernal_size = kernel_size
        self.dilation = dilation


        self.conv_1 = weight_norm(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation))
        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(dropout)
        
        self.conv_2 = weight_norm(nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation))
        self.relu_2 = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

        # One block consists of two structures conv->relu->dropout
        self.nn_block1 = nn.Sequential(
            self.conv_1,
            self.relu_1,
            self.dropout_1
        )

        self.nn_block2 = nn.Sequential(
            self.conv_2,
            self.relu_2,
            self.dropout_2
        )

        # Use 1x1 convolution to use the residual connection
        self.conv_1x1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1) if in_channels != out_channels else None # 1x1 convolution is not needed if the number of channels has not changed

        self.init_weights()

    # Normal initialization of weights
    def init_weights(self):
        self.conv_1.weight.data.normal_(0, 0.01)
        self.conv_2.weight.data.normal_(0, 0.01)
        if self.conv_1x1 is not None:
            self.conv_1x1.weight.data.normal_(0, 0.01)

    # To use causal convolution, expand the object with zeros only on the left
    def augment(self, X, padding):
        addition = torch.zeros((X.shape[0],X.shape[1],padding))
        X_aug = torch.cat((addition, X),2)
        return X_aug
    # Forward pass
    def forward(self, X):
        left_padding = self.dilation * (self.kernal_size-1)
        X_aug = self.augment(X, left_padding)

        out = self.nn_block1(X_aug)
        out_aug = self.augment(out, left_padding)
        out = self.nn_block2(out_aug)
        
        out_X = self.conv_1x1(X) if self.conv_1x1 is not None else X # if the 1x1 сonv layer is not defined, do not change the output

        return out_X + out # use the residual connection

# A class for creating a TCN architecture
class TCN(nn.Module):
    def __init__(self, channels_list, kernel_size, dropout=0):
        """channels_list - list the depth of each layer in order"""
        super(TCN, self).__init__()
        layers = []
        for i in range(len(channels_list)-1):
            dilation = 2**i
            layers.append(ResidualBlock(channels_list[i], channels_list[i+1], kernel_size, dilation, dropout))
        
        self.TCN_model = nn.Sequential(*layers)

    def forward(self, X):
        return self.TCN_model(X)


    