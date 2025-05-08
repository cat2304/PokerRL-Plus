# Copyright (c) 2019 Eric Steinberger


import torch.nn as nn

from PokerRL.rl.neural.MainPokerModuleFLAT import MainPokerModuleFLAT
from PokerRL.rl.neural.MainPokerModuleRNN import MainPokerModuleRNN


class NeuralNet(nn.Module):
    """
    神经网络基类，用于所有强化学习算法。
    该网络接收观察值作为输入，并输出相应的值（如Q值、策略分布等）。
    网络可以使用RNN或扁平结构来处理观察序列。
    这个类提供了基本的网络结构和前向传播功能。
    """

    def __init__(self, env_bldr, net_args, device):
        """
        参数：
            env_bldr:      PokerEnvBuilder实例，用于构建和管理扑克环境
            net_args:      包含所有设置的参数对象
            device:        torch设备（CPU或GPU）
        """
        super().__init__()

        self.args = net_args

        if net_args.use_rnn:
            self.main_net = MainPokerModuleRNN(env_bldr=env_bldr, device=device, mpm_args=net_args.mpm_args)
        else:
            self.main_net = MainPokerModuleFLAT(env_bldr=env_bldr, device=device, mpm_args=net_args.mpm_args)

        self.output_net = nn.Sequential(
            nn.Linear(self.main_net.output_units, net_args.n_units_final),
            nn.ReLU(),
            nn.Linear(net_args.n_units_final, net_args.output_dim)
        )

    def forward(self, obs):
        """
        前向传播函数。

        参数：
            obs:    观察值，包含游戏状态信息

        返回：
            网络输出，形状由output_dim决定
            具体输出类型取决于子类的实现
        """
        y = self.main_net(obs=obs)
        return self.output_net(y) 