# Copyright (c) 2019 Eric Steinberger


import torch.nn as nn

from PokerRL.rl.neural.MainPokerModuleFLAT import MainPokerModuleFLAT
from PokerRL.rl.neural.MainPokerModuleRNN import MainPokerModuleRNN


class AdvantageNet(nn.Module):
    """
    优势网络，用于Actor-Critic算法。
    该网络计算每个动作相对于平均动作值的优势。
    优势值表示某个动作比平均动作"好多少"。
    这种设计有助于减少策略梯度的方差，提高训练稳定性。
    网络可以使用RNN或扁平结构来处理观察序列。
    """

    def __init__(self, env_bldr, adv_args, device):
        """
        参数：
            env_bldr:      PokerEnvBuilder实例，用于构建和管理扑克环境
            adv_args:      包含所有设置的参数对象
            device:        torch设备（CPU或GPU）
        """
        super().__init__()

        self.args = adv_args
        self.n_actions = env_bldr.N_ACTIONS

        if adv_args.use_rnn:
            self.main_net = MainPokerModuleRNN(env_bldr=env_bldr, device=device, mpm_args=adv_args.mpm_args)
        else:
            self.main_net = MainPokerModuleFLAT(env_bldr=env_bldr, device=device, mpm_args=adv_args.mpm_args)

        self.advantage_net = nn.Sequential(
            nn.Linear(self.main_net.output_units, adv_args.n_units_final),
            nn.ReLU(),
            nn.Linear(adv_args.n_units_final, self.n_actions)
        )

    def forward(self, obs):
        """
        前向传播函数。

        参数：
            obs:    观察值，包含游戏状态信息

        返回：
            每个动作的优势值，形状为 [batch_size, n_actions]
            优势值表示该动作比平均动作的预期收益高多少
        """
        y = self.main_net(obs=obs)
        return self.advantage_net(y)


class AdvantageArgs:
    """
    优势网络的参数类
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
