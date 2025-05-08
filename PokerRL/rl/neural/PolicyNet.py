# Copyright (c) 2019 Eric Steinberger


import torch.nn as nn

from PokerRL.rl.neural.MainPokerModuleFLAT import MainPokerModuleFLAT
from PokerRL.rl.neural.MainPokerModuleRNN import MainPokerModuleRNN


class PolicyNet(nn.Module):
    """
    策略网络，用于策略梯度算法。
    该网络接收观察值作为输入，并输出每个动作的概率分布。
    网络可以使用RNN或扁平结构来处理观察序列。
    输出层使用softmax激活函数，确保输出是一个有效的概率分布。
    """

    def __init__(self, env_bldr, policy_args, device):
        """
        参数：
            env_bldr:      PokerEnvBuilder实例，用于构建和管理扑克环境
            policy_args:   包含所有设置的参数对象
            device:        torch设备（CPU或GPU）
        """
        super().__init__()

        self.args = policy_args
        self.n_actions = env_bldr.N_ACTIONS

        if policy_args.use_rnn:
            self.main_net = MainPokerModuleRNN(env_bldr=env_bldr, device=device, mpm_args=policy_args.mpm_args)
        else:
            self.main_net = MainPokerModuleFLAT(env_bldr=env_bldr, device=device, mpm_args=policy_args.mpm_args)

        self.policy_net = nn.Sequential(
            nn.Linear(self.main_net.output_units, policy_args.n_units_final),
            nn.ReLU(),
            nn.Linear(policy_args.n_units_final, self.n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs):
        """
        前向传播函数。

        参数：
            obs:    观察值，包含游戏状态信息

        返回：
            每个动作的概率分布，形状为 [batch_size, n_actions]
            概率分布已经通过softmax归一化，确保所有动作的概率和为1
        """
        y = self.main_net(obs=obs)
        return self.policy_net(y) 