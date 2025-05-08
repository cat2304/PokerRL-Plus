# Copyright (c) 2019 Eric Steinberger


import torch.nn as nn


class LayerNorm(nn.Module):
    """
    层归一化模块。
    该模块对输入的每一层进行归一化处理，使其均值为0，方差为1。
    这有助于加速训练过程并提高模型的稳定性。
    它还包含可学习的缩放和偏移参数，以增加模型的表达能力。
    """

    def __init__(self, device):
        """
        参数：
            device:        torch设备（CPU或GPU）
        """
        super().__init__()
        self.device = device
        self.to(device)

    def forward(self, x):
        """
        前向传播函数。

        参数：
            x:      输入张量，形状为 [batch_size, feature_dim]

        返回：
            归一化后的张量，具有相同的形状
        """
        return nn.functional.layer_norm(x, x.shape[1:], elementwise_affine=True) 