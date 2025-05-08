# Copyright (c) 2019 Eric Steinberger


import torch.nn as nn


class HighwayLayer(nn.Module):
    """
    高速网络层。
    这是一种特殊的神经网络层，它允许信息直接通过或经过变换。
    它包含一个变换门和一个携带门，用于控制信息流动。
    这种设计有助于训练非常深的网络，因为它可以缓解梯度消失问题。
    """

    def __init__(self, dim):
        """
        参数：
            dim:    输入和输出的维度
        """
        super().__init__()

        self.H = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU()
        )
        self.T = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        前向传播函数。

        参数：
            x:      输入张量，形状为 [batch_size, dim]

        返回：
            经过高速层处理的张量，形状为 [batch_size, dim]
            输出是变换分支和直通分支的加权组合
        """
        H = self.H(x)
        T = self.T(x)
        return H * T + x * (1.0 - T) 