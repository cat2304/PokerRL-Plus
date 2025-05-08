# Copyright (c) 2019 Eric Steinberger


import torch.nn as nn

from PokerRL.rl import rl_util


class NetWrapperBase:
    """
    神经网络包装器的基类。
    它提供了一些基本功能，如加载和保存模型、切换训练/评估模式等。
    所有具体的网络包装器都应该继承这个类，并根据需要重写相关方法。
    这个类的设计遵循了组合模式，将网络的通用功能封装在基类中。
    """

    def __init__(self, net:nn.Module, env_bldr, args, owner, device):
        """
        参数：
            net:            神经网络模块，实际的网络实现
            env_bldr:      PokerEnvBuilder实例，用于构建和管理扑克环境
            args:          包含所有设置的参数对象
            owner:         该网络所属的玩家的seat_id
            device:        torch设备（CPU或GPU）
        """
        self.net = net
        self.env_bldr = env_bldr
        self.args = args
        self.owner = owner
        self.device = device

        self._criterion = rl_util.str_to_loss_cls(self.args.loss_str)
        self.loss_last_batch = None

    def get_grads_one_batch_from_buffer(self, buffer):
        if buffer.size < self.args.batch_size:
            return

        self.net.train()
        # 可能需要进行多个小批量处理，因为所有数据可能无法一次性放入每个GPU的显存中。
        _grad_mngr = _GradManager(net=self.net, args=self.args, criterion=self._criterion)
        for micro_batch_id in range(self.args.n_mini_batches_per_update):
            self._mini_batch_loop(buffer=buffer, grad_mngr=_grad_mngr)

        self.loss_last_batch = _grad_mngr.get_loss_sum()
        return _grad_mngr.average()

    def _mini_batch_loop(self, buffer, grad_mngr):
        raise NotImplementedError

    def get_weights(self):
        """
        获取网络的权重。

        返回：
            字典，包含网络的所有参数（权重和偏置）
        """
        return self.net.state_dict()

    def load_net_state_dict(self, state_dict):
        """
        加载网络的权重。

        参数：
            state_dict:      字典，包含要加载的网络参数
        """
        self.net.load_state_dict(state_dict)

    def evaluate(self):
        """
        将网络设置为评估模式。
        在评估模式下，某些层（如dropout和batch normalization）会有不同的行为。
        这个模式通常用于测试和推理阶段。
        """
        self.net.eval()

    def train(self):
        """
        将网络设置为训练模式。
        在训练模式下，所有层都会正常工作，包括dropout和batch normalization。
        这个模式用于网络的训练阶段。
        """
        self.net.train()

    def to(self, device):
        """
        将网络移动到指定设备。

        参数：
            device:        torch设备（CPU或GPU），用于指定网络运行的硬件
        """
        self.device = device
        self.net.to(device)

    def state_dict(self):
        """ 
        获取网络的状态字典。
        如果需要自定义保存的内容，子类可以重写此方法。
        """
        return self.get_weights()

    def load_state_dict(self, state):
        """ 
        加载网络的状态字典。
        如果需要自定义加载的行为，子类可以重写此方法。
        """
        self.load_net_state_dict(state)


class NetWrapperArgsBase:

    def __init__(self,
                 batch_size,
                 n_mini_batches_per_update,
                 optim_str,
                 loss_str,
                 lr,
                 grad_norm_clipping,
                 device_training
                 ):
        assert isinstance(device_training, str), "Please pass a string (either 'cpu' or 'cuda')!"
        self.batch_size = batch_size
        self.n_mini_batches_per_update = n_mini_batches_per_update
        self.optim_str = optim_str
        self.loss_str = loss_str
        self.lr = lr
        self.grad_norm_clipping = grad_norm_clipping
        self.device_training = torch.device(device_training)


class _GradManager:

    def __init__(self, args, net, criterion):
        self.net = net
        self.args = args
        self.criterion = criterion
        self._grads = {}
        self._loss_sum = 0.0
        for name, _ in net.named_parameters():
            self._grads[name] = []

    def backprop(self, pred, target, loss_weights=None):
        self.net.zero_grad()
        if loss_weights is None:
            loss = self.criterion(pred, target)
        else:
            loss = self.criterion(pred, target, loss_weights)
        loss.backward()
        self._loss_sum += loss.item()
        self.add()

    def add(self):
        for name, param in self.net.named_parameters():
            self._grads[name].append(param.grad.data.clone())

    def average(self):
        if self.args.n_mini_batches_per_update == 1:
            for name, param in self.net.named_parameters():
                self._grads[name] = self._grads[name][0]
        else:
            for name, param in self.net.named_parameters():
                self._grads[name] = torch.mean(torch.stack(self._grads[name], dim=0), dim=0)
        return self._grads

    def get_loss_sum(self):
        return self._loss_sum
