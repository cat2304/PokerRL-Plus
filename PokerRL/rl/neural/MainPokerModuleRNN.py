# Copyright (c) 2019 Eric Steinberger


import torch
import torch.nn as nn

from PokerRL.rl import rl_util
from PokerRL.rl.neural.CardEmbedding import CardEmbedding
from PokerRL.rl.neural.LayerNorm import LayerNorm


class MainPokerModuleRNN(nn.Module):
    """
    主扑克模块的RNN版本。
    该模块使用RNN处理观察序列，能够捕捉时序依赖关系。
    它包含了卡牌嵌入、状态编码和RNN层等组件。
    这个模块是所有基于RNN的扑克AI网络的基础。
    """

    def __init__(self,
                 env_bldr,
                 device,
                 mpm_args,
                 ):
        """
        参数：
            env_bldr:      PokerEnvBuilder实例，用于构建和管理扑克环境
            device:        torch设备（CPU或GPU）
            mpm_args:      主扑克模块的参数对象
        """
        super().__init__()
        self.args = mpm_args

        self.env_bldr = env_bldr

        self.N_SEATS = self.env_bldr.N_SEATS
        self.device = device

        self.board_start = self.env_bldr.obs_board_idxs[0]
        self.board_len = len(self.env_bldr.obs_board_idxs)

        self.table_start = self.env_bldr.obs_table_state_idxs[0]
        self.table_len = len(self.env_bldr.obs_table_state_idxs)

        self.players_info_starts = [player_i_idxs[0] for player_i_idxs in self.env_bldr.obs_players_idxs]
        self.players_info_lens = [len(player_i_idxs) for player_i_idxs in self.env_bldr.obs_players_idxs]

        self.pub_obs_size = self.env_bldr.pub_obs_size
        self.priv_obs_size = self.env_bldr.priv_obs_size

        self._relu = nn.ReLU(inplace=False)

        self.card_emb = CardEmbedding(env_bldr=env_bldr, dim=mpm_args.card_emb_dim, device=device)
        self.norm_inp = LayerNorm(device=device)

        self.rnn = nn.GRU(input_size=self._input_size(),
                         hidden_size=mpm_args.rnn_units,
                         num_layers=mpm_args.rnn_layers,
                         batch_first=True)

        self.output_units = mpm_args.rnn_units

        self.lut_range_idx_2_priv_o = torch.from_numpy(self.env_bldr.lut_holder.LUT_RANGE_IDX_TO_PRIVATE_OBS)
        self.lut_range_idx_2_priv_o = self.lut_range_idx_2_priv_o.to(device=self.device, dtype=torch.float32)

        self.to(device)

    def forward(self, pub_obses, range_idxs):
        """
        参数：
            pub_obses (list):                 形状为[np.arr([history_len, n_features]), ...)的numpy数组列表
            range_idxs (LongTensor):         每个pub_obs对应的range_idxs张量([2, 421, 58, 912, ...])

        返回：
            RNN处理后的特征，包含了历史信息的时序表示
        """
        # 获取私有观察
        priv_obs = self.lut_range_idx_2_priv_o[range_idxs]

        if len(pub_obses) > 1:
            # 处理批量数据
            seq_lens = torch.tensor([sample.shape[0] for sample in pub_obses], device=self.device, dtype=torch.int32)
            max_len = seq_lens.max().item()

            # 将公共观察转换为张量
            _pub_obs = pub_obses
            pub_obses = torch.zeros((max_len, len(pub_obses), self.pub_obs_size), dtype=torch.float32, device=self.device)
            for i, pub in zip(range(len(_pub_obs)), _pub_obs):
                pub_obses[:seq_lens[i], i] = torch.from_numpy(pub).to(self.device)

            # 扩展私有观察到序列长度
            priv_obs = priv_obs.unsqueeze(0).repeat(max_len, 1, 1)

            # 通过预处理层
            y = self._feed_through_pre_layers(pub_o=pub_obses, priv_o=priv_obs)

            # 按序列长度降序排序
            seq_lens, idx_shifts = torch.sort(seq_lens, descending=True)
            y = y[:, idx_shifts, :]

            # 打包序列
            y = torch.nn.utils.rnn.pack_padded_sequence(y, lengths=seq_lens, batch_first=False)

            # 通过RNN
            y, _ = self.rnn(y)

            # 解包序列
            y, seq_lens = nn.utils.rnn.pad_packed_sequence(y, batch_first=False)

            if self.args.sum_step_outputs:
                # 对序列求平均
                y = y.sum(0) * (1.0 / seq_lens.float()).unsqueeze(-1)
            else:
                # 使用序列的最后一个输出
                y = y[seq_lens - 1, torch.arange(len(pub_obses), device=self.device, dtype=torch.long), :].squeeze(dim=0)

            # 恢复原始顺序
            idx_unsort_obs_t = torch.arange(len(pub_obses), device=self.device, dtype=torch.long)
            idx_unsort_obs_t.scatter_(src=idx_unsort_obs_t.clone(), dim=0, index=idx_shifts)

            return y[idx_unsort_obs_t]

        else:
            # 处理单个序列
            seq_len = pub_obses[0].shape[0]
            pub_obses = torch.from_numpy(pub_obses[0]).to(self.device).view(seq_len, 1, self.pub_obs_size)
            priv_obs = priv_obs.unsqueeze(0).expand(seq_len, 1, self.priv_obs_size)

            # 通过预处理层
            y = self._feed_through_pre_layers(pub_o=pub_obses, priv_o=priv_obs)

            y, _ = self.rnn(y)

            if self.args.sum_step_outputs:
                # 对序列求平均
                return y.sum(0) * (1.0 / seq_len)
            else:
                # 使用序列的最后一个输出
                return y[seq_len - 1].view(1, -1)

    def _feed_through_pre_layers(self, pub_o, priv_o):
        """
        通过预处理层处理观察。

        参数：
            pub_o:      公共观察
            priv_o:     私有观察

        返回：
            处理后的特征
        """
        # 牌局主体
        _cards_obs = torch.cat((priv_o, pub_o.narrow(dim=-1, start=self.board_start, length=self.board_len)), dim=-1)
        cards_out = self._relu(self.cards_fc_1(_cards_obs))
        cards_out = self._relu(self.cards_fc_2(cards_out) + cards_out)
        cards_out = self._relu(self.cards_fc_3(cards_out) + cards_out)

        # 桌面主体
        _table_obs = torch.cat(
            [
                pub_o.narrow(dim=-1, start=self.table_start, length=self.table_len)
            ]
            +
            [
                pub_o.narrow(dim=-1, start=self.players_info_starts[i],
                           length=self.players_info_lens[i])
                for i in range(self.N_SEATS)
            ]
            , dim=-1
        )
        table_out = self._relu(self.table_state_fc(_table_obs))

        # 合并层
        return self._relu(self.merge_fc(torch.cat([cards_out, table_out], dim=-1)))

    def _input_size(self):
        """
        计算RNN输入维度。

        返回：
            输入特征的总维度
        """
        return self.card_emb.out_size * (1 + self.env_bldr.N_CARDS_BOARD) + self.env_bldr.N_ACTIONS


class MPMArgsRNN:
    """
    RNN神经网络模块的参数类
    """

    def __init__(self,
                 rnn_units=128,
                 rnn_stack=2,
                 rnn_dropout=0.0,
                 rnn_cls_str="lstm",
                 use_pre_layers=True,
                 n_cards_state_units=192,
                 n_merge_and_table_layer_units=64,
                 sum_step_outputs=False):
        """
        参数：
            rnn_units:                      RNN隐藏单元数量
            rnn_stack:                      RNN层数
            rnn_dropout:                    RNN dropout率
            rnn_cls_str:                   RNN类型（"lstm"或"gru"）
            use_pre_layers:                是否使用预处理层
            n_cards_state_units:           牌局状态单元数量
            n_merge_and_table_layer_units: 合并和桌面层单元数量
            sum_step_outputs:              是否对序列输出求平均
        """
        self.rnn_units = rnn_units
        self.rnn_stack = rnn_stack
        self.rnn_dropout = rnn_dropout
        self.rnn_cls_str = rnn_cls_str
        self.use_pre_layers = use_pre_layers
        self.n_cards_state_units = n_cards_state_units
        self.n_merge_and_table_layer_units = n_merge_and_table_layer_units
        self.sum_step_outputs = sum_step_outputs

    def get_mpm_cls(self):
        return MainPokerModuleRNN
