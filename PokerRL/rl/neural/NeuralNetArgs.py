# Copyright (c) 2019 Eric Steinberger


from PokerRL.rl.neural.MainPokerModuleFLAT import MPMArgsFLAT
from PokerRL.rl.neural.MainPokerModuleRNN import MPMArgsRNN


class NeuralNetArgs:
    """
    神经网络参数类，用于配置神经网络的超参数。
    该类包含了所有神经网络相关的配置选项，如是否使用RNN、网络层大小等。
    这些参数会影响网络的架构和训练过程。
    """

    def __init__(self,
                 use_rnn=False,
                 n_units_final=64,
                 output_dim=1,
                 mpm_args=None):
        """
        参数：
            use_rnn:        是否使用RNN结构，默认为False
            n_units_final:  最终层的神经元数量，默认为64
            output_dim:     输出维度，默认为1
            mpm_args:       主扑克模块的参数，如果为None则使用默认参数
        """
        self.use_rnn = use_rnn
        self.n_units_final = n_units_final
        self.output_dim = output_dim

        if mpm_args is None:
            if use_rnn:
                self.mpm_args = MPMArgsRNN()
            else:
                self.mpm_args = MPMArgsFLAT()
        else:
            self.mpm_args = mpm_args 