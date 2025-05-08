# Copyright (c) 2019 Eric Steinberger


import torch.nn as nn

from PokerRL.rl.neural.MainPokerModuleFLAT import MainPokerModuleFLAT
from PokerRL.rl.neural.MainPokerModuleRNN import MainPokerModuleRNN


class QNet(nn.Module):
    """
    Q网络，用于深度Q学习。
    该网络接收观察值作为输入，并输出每个动作的Q值。
    Q值代表在给定状态下采取某个动作的预期累积奖励。
    网络可以使用RNN或扁平结构来处理观察序列。
    """

    def __init__(self, env_bldr, q_args, device):
        """
        参数：
            env_bldr:      PokerEnvBuilder实例，用于构建和管理扑克环境
            q_args:        包含所有设置的参数对象
            device:        torch设备（CPU或GPU）
        """
        super().__init__()

        self.args = q_args
        self.n_actions = env_bldr.N_ACTIONS

        if q_args.use_rnn:
            self.main_net = MainPokerModuleRNN(env_bldr=env_bldr, device=device, mpm_args=q_args.mpm_args)
        else:
            self.main_net = MainPokerModuleFLAT(env_bldr=env_bldr, device=device, mpm_args=q_args.mpm_args)

        self.value_net = nn.Sequential(
            nn.Linear(self.main_net.output_units, q_args.n_units_final),
            nn.ReLU(),
            nn.Linear(q_args.n_units_final, self.n_actions)
        )

    def forward(self, obs):
        """
        前向传播函数。

        参数：
            obs:    观察值，包含游戏状态信息

        返回：
            每个动作的Q值，形状为 [batch_size, n_actions]
        """
        y = self.main_net(obs=obs)
        return self.value_net(y)


class QArgs:
    """
    Q网络的参数类
    """

    def __init__(self,
                 mpm_args,
                 n_units_final=64):
        """
        参数：
            mpm_args:           MainPokerModule的参数对象
            n_units_final:      最后一层的神经元数量
        """
        self.mpm_args = mpm_args
        self.n_units_final = n_units_final
