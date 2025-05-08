# Copyright (c) 2019 Eric Steinberger


import torch.nn as nn


class BaselineNet(nn.Module):
    """
    一个基线网络，用于策略梯度算法。
    它将观察作为输入，并输出一个标量值作为基线。
    该基线用于减少策略梯度的方差，通过从回报中减去基线值来实现。
    这种方法可以显著提高策略梯度算法的稳定性和收敛性。
    """

    def __init__(self, env_bldr, baseline_args, device):
        """
        参数：
            env_bldr:          PokerEnvBuilder实例
            baseline_args:     包含所有设置的参数对象
            device:            torch设备
        """
        super().__init__()

        self.args = baseline_args
        self.env_bldr = env_bldr

        MPM = baseline_args.mpm_args.get_mpm_cls()
        self.main_net = MPM(env_bldr=env_bldr, device=device, mpm_args=baseline_args.mpm_args)

        self.value = nn.Linear(in_features=self.main_net.output_units,
                             out_features=1)

        self.to(device)

    def forward(self, pub_obses, range_idxs):
        """
        参数：
            pub_obses (list):                 形状为[np.arr([history_len, n_features]), ...)的numpy数组列表
            range_idxs (LongTensor):         每个pub_obs对应的range_idxs张量([2, 421, 58, 912, ...])

        返回：
            基线值（标量），用于从回报中减去以减少策略梯度的方差
        """
        y = self.main_net(pub_obses=pub_obses, range_idxs=range_idxs)
        baseline = self.value(y)
        return baseline


class BaselineArgs:
    """
    基线网络的参数类
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