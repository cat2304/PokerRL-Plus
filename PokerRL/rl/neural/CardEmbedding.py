# Copyright (c) 2019 Eric Steinberger


import torch.nn as nn


class CardEmbedding(nn.Module):
    """
    卡牌嵌入模块。
    该模块使用嵌入层将离散的卡牌索引映射到连续的向量空间。
    这种表示方式可以捕捉卡牌之间的相似性和关系。
    嵌入向量的维度是可配置的，以适应不同的需求。
    """

    def __init__(self, env_bldr, dim, device):
        """
        参数：
            env_bldr:      PokerEnvBuilder实例，用于构建和管理扑克环境
            dim:           嵌入向量的维度
            device:        torch设备（CPU或GPU）
        """
        super().__init__()

        self.env_bldr = env_bldr
        self.dim = dim
        self.device = device

        self.card_emb = nn.Embedding(
            num_embeddings=env_bldr.N_CARDS_IN_DECK,
            embedding_dim=dim
        )

        self.to(device)

    @property
    def out_size(self):
        """
        返回嵌入向量的维度。
        """
        return self.dim

    def forward(self, card_idxs):
        """
        前向传播函数。

        参数：
            card_idxs:     卡牌索引张量，形状为 [batch_size, n_cards]

        返回：
            卡牌的嵌入向量，形状为 [batch_size, n_cards, dim]
        """
        return self.card_emb(card_idxs)


class CardEmbeddingArgs:
    """
    卡牌嵌入网络的参数类
    """

    def __init__(self,
                 dim=64):
        """
        参数：
            dim:           嵌入向量的维度，默认为64
        """
        self.dim = dim 