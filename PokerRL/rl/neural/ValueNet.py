# Copyright (c) 2019 Eric Steinberger


import torch.nn as nn

from PokerRL.rl.neural.MainPokerModuleFLAT import MainPokerModuleFLAT
from PokerRL.rl.neural.MainPokerModuleRNN import MainPokerModuleRNN


class ValueNet(nn.Module):
    """
    价值网络，用于Actor-Critic算法。
    该网络接收观察值作为输入，并输出状态价值（一个标量值）。
    状态价值表示从当前状态开始，遵循当前策略所能获得的预期累积奖励。
    网络可以使用RNN或扁平结构来处理观察序列。
    """

    def __init__(self, env_bldr, value_args, device):
        """
        参数：
            env_bldr:      PokerEnvBuilder实例，用于构建和管理扑克环境
            value_args:    包含所有设置的参数对象
            device:        torch设备（CPU或GPU）
        """
        super().__init__()

        self.args = value_args

        if value_args.use_rnn:
            self.main_net = MainPokerModuleRNN(env_bldr=env_bldr, device=device, mpm_args=value_args.mpm_args)
        else:
            self.main_net = MainPokerModuleFLAT(env_bldr=env_bldr, device=device, mpm_args=value_args.mpm_args)

        self.value_net = nn.Sequential(
            nn.Linear(self.main_net.output_units, value_args.n_units_final),
            nn.ReLU(),
            nn.Linear(value_args.n_units_final, 1)
        )

    def forward(self, obs):
        """
        前向传播函数。

        参数：
            obs:    观察值，包含游戏状态信息

        返回：
            状态价值（标量），形状为 [batch_size, 1]
            表示从当前状态开始，遵循当前策略所能获得的预期累积奖励
        """
        y = self.main_net(obs=obs)
        return self.value_net(y) 