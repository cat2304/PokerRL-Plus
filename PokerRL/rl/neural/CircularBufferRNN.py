# Copyright (c) 2019 Eric Steinberger


import numpy as np
import torch


class CircularBufferRNN:
    """
    循环缓冲区，用于存储RNN的隐藏状态和观察序列。
    该类使用循环数组实现固定大小的缓冲区，当缓冲区满时，新数据会覆盖最旧的数据。
    这种设计可以高效地管理内存，适用于需要存储大量序列数据的场景。
    """

    def __init__(self, max_size):
        """
        参数：
            max_size:   缓冲区的最大容量
        """
        self.max_size = max_size
        self.size = 0
        self.current_idx = 0
        self.states = [None for _ in range(max_size)]
        self.obs_sequences = [None for _ in range(max_size)]

    def add(self, state, obs_sequence):
        """
        添加新的状态和观察序列到缓冲区。
        如果缓冲区已满，新数据会覆盖最旧的数据。

        参数：
            state:          RNN的隐藏状态
            obs_sequence:   对应的观察序列
        """
        self.states[self.current_idx] = state
        self.obs_sequences[self.current_idx] = obs_sequence

        self.current_idx = (self.current_idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def get_random(self, batch_size):
        """
        随机获取一批数据。

        参数：
            batch_size:     要获取的数据数量

        返回：
            states:         RNN隐藏状态的列表
            obs_sequences:  对应的观察序列列表
        """
        indices = np.random.randint(low=0, high=self.size, size=batch_size)
        return [self.states[idx] for idx in indices], [self.obs_sequences[idx] for idx in indices]

    def sample(self, batch_size, device):
        """
        从缓冲区中随机采样一批数据。

        参数：
            batch_size:     批量大小
            device:        torch设备

        返回：
            pub_obs_batch:     公共观察批量
            range_idx_batch:   范围索引批量
            hidden_batch:      隐藏状态批量
        """
        indices = np.random.randint(low=0, high=self.size, size=batch_size)

        pub_obs_batch = [self.pub_obs_buffer[i] for i in indices]
        range_idx_batch = torch.from_numpy(self.range_idx_buffer[indices]).to(device)
        hidden_batch = [self.hidden_states[i] for i in indices]

        return pub_obs_batch, range_idx_batch, hidden_batch 