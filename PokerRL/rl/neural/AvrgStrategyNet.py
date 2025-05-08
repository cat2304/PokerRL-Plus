# Copyright (c) 2019 Eric Steinberger


import torch
import torch.nn as nn


class AvrgStrategyNet(nn.Module):
    """
    一个平均策略网络，用于CFR算法。
    它将观察作为输入，并输出每个动作的概率分布（未经过softmax）。
    该网络用于近似计算CFR中的平均策略，帮助策略收敛到纳什均衡。
    """

    def __init__(self, env_bldr, avrg_net_args, device):
        """
        参数：
            env_bldr:          PokerEnvBuilder实例
            avrg_net_args:     包含所有设置的参数对象
            device:            torch设备
        """
        super().__init__()

        self.args = avrg_net_args
        self.env_bldr = env_bldr
        self.n_actions = env_bldr.N_ACTIONS

        MPM = avrg_net_args.mpm_args.get_mpm_cls()
        self.main_net = MPM(env_bldr=env_bldr, device=device, mpm_args=avrg_net_args.mpm_args)

        self.value = nn.Linear(in_features=self.main_net.output_units,
                             out_features=self.n_actions)

        self.to(device)

    def forward(self, pub_obses, range_idxs):
        """
        参数：
            pub_obses (list):                 形状为[np.arr([history_len, n_features]), ...)的numpy数组列表
            range_idxs (LongTensor):         每个pub_obs对应的range_idxs张量([2, 421, 58, 912, ...])

        返回：
            每个动作的概率分布的logits（未经过softmax），需要通过softmax转换为概率
        """
        y = self.main_net(pub_obses=pub_obses, range_idxs=range_idxs)
        logits = self.value(y)
        return logits


class AvrgNetArgs:
    """
    平均策略网络的参数类
    """

    def __init__(self,
                 mpm_args,
                 n_units_final=64):
        """
        参数：
            mpm_args:           MainPokerModule的参数对象
            n_units_final:      最后一层的神经元数量，默认为64
        """
        self.mpm_args = mpm_args
        self.n_units_final = n_units_final
