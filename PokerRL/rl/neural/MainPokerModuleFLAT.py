# Copyright (c) 2019 Eric Steinberger


import torch.nn as nn

from PokerRL.rl.neural.CardEmbedding import CardEmbedding
from PokerRL.rl.neural.LayerNorm import LayerNorm


class MainPokerModuleFLAT(nn.Module):
    """
    主扑克模块的扁平版本。
    该模块使用全连接层处理观察值，不包含时序处理能力。
    它包含了卡牌嵌入、状态编码等组件。
    这个模块是所有基于前馈网络的扑克AI网络的基础。
    """

    def __init__(self, env_bldr, device, mpm_args):
        """
        参数：
            env_bldr:      PokerEnvBuilder实例，用于构建和管理扑克环境
            device:        torch设备（CPU或GPU）
            mpm_args:      主扑克模块的参数对象
        """
        super().__init__()

        self.args = mpm_args
        self.env_bldr = env_bldr
        self.device = device

        self.card_emb = CardEmbedding(env_bldr=env_bldr, dim=mpm_args.card_emb_dim, device=device)
        self.norm_inp = LayerNorm(device=device)

        self.main_net = nn.Sequential(
            nn.Linear(self._input_size(), mpm_args.hidden_units),
            nn.ReLU(),
            nn.Linear(mpm_args.hidden_units, mpm_args.hidden_units),
            nn.ReLU()
        )

        self.output_units = mpm_args.hidden_units

    def forward(self, obs):
        """
        前向传播函数。

        参数：
            obs:    观察值，包含游戏状态信息

        返回：
            处理后的特征向量
        """
        # 处理输入
        y = self._input_net(obs)
        y = self.norm_inp(y)

        # 前馈网络处理
        return self.main_net(y)

    def _input_size(self):
        """
        计算网络输入维度。

        返回：
            输入特征的总维度
        """
        return self.card_emb.out_size * (1 + self.env_bldr.N_CARDS_BOARD) + self.env_bldr.N_ACTIONS


class MPMArgsFLAT:
    """
    扁平神经网络模块的参数类
    """

    def __init__(self,
                 n_cards_state_units=192,
                 n_merge_and_table_layer_units=64,
                 use_pre_layers=True):
        """
        参数：
            n_cards_state_units:              牌局状态单元数量，用于处理牌的信息
            n_merge_and_table_layer_units:    合并和桌面层单元数量，用于处理综合特征
            use_pre_layers:                   是否使用预处理层来分别处理不同类型的特征
        """
        self.n_cards_state_units = n_cards_state_units
        self.n_merge_and_table_layer_units = n_merge_and_table_layer_units
        self.use_pre_layers = use_pre_layers

    def get_mpm_cls(self):
        return MainPokerModuleFLAT
