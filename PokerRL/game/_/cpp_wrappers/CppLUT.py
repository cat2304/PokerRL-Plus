# Copyright (c) 2019 Eric Steinberger


import os
from os.path import join as ospj

import numpy as np

from PokerRL._.CppWrapper import CppWrapper
from PokerRL.game.Poker import Poker
from PokerRL.game._.rl_env.game_rules import HoldemRules


class CppLibHoldemLuts(CppWrapper):
    """
    德州扑克查找表(LUT)的C++库包装器
    
    这个类继承自CppWrapper，用于管理德州扑克游戏中的各种查找表。
    主要功能包括：
    1. 手牌索引和实际手牌之间的转换
    2. 公共牌（翻牌、转牌、河牌）的索引管理
    3. 一维和二维卡牌表示之间的转换
    
    使用示例：
    ```python
    # 创建查找表管理器
    lut_manager = CppLibHoldemLuts(n_boards_lut={...}, n_cards_out_lut={...})
    
    # 获取手牌索引到实际手牌的映射
    idx_to_hole = lut_manager.get_idx_2_hole_card_lut()
    
    # 获取实际手牌到索引的映射
    hole_to_idx = lut_manager.get_hole_card_2_idx_lut()
    ```
    """

    def __init__(self, n_boards_lut, n_cards_out_lut):
        """
        初始化查找表管理器
        
        参数:
            n_boards_lut (dict): 每个游戏阶段可能的公共牌组合数量
            n_cards_out_lut (dict): 每个游戏阶段需要发出的牌数
        """
        # 加载动态链接库
        super().__init__(path_to_dll=ospj(os.path.dirname(os.path.realpath(__file__)),
                                          "lib_luts." + self.CPP_LIB_FILE_ENDING))
        self._n_boards_lut = n_boards_lut
        self._n_cards_out_lut = n_cards_out_lut

        # 设置各个查找表函数的参数类型
        self._clib.get_hole_card_2_idx_lut.argtypes = [self.ARR_2D_ARG_TYPE]
        self._clib.get_hole_card_2_idx_lut.restype = None

        self._clib.get_idx_2_hole_card_lut.argtypes = [self.ARR_2D_ARG_TYPE]
        self._clib.get_idx_2_hole_card_lut.restype = None

        self._clib.get_idx_2_flop_lut.argtypes = [self.ARR_2D_ARG_TYPE]
        self._clib.get_idx_2_flop_lut.restype = None

        self._clib.get_idx_2_turn_lut.argtypes = [self.ARR_2D_ARG_TYPE]
        self._clib.get_idx_2_turn_lut.restype = None

        self._clib.get_idx_2_river_lut.argtypes = [self.ARR_2D_ARG_TYPE]
        self._clib.get_idx_2_river_lut.restype = None

    # __________________________________________________ LUTs __________________________________________________________
    def get_idx_2_hole_card_lut(self):
        """
        获取手牌索引到实际手牌的查找表
        
        返回:
            np.ndarray: 形状为[RANGE_SIZE, 2]的数组
                - RANGE_SIZE是所有可能的手牌组合数量
                - 每行表示一对手牌的点数和花色
                - -2表示无效值
        """
        lut = np.full(shape=(HoldemRules.RANGE_SIZE, 2), fill_value=-2, dtype=np.int8)
        self._clib.get_idx_2_hole_card_lut(self.np_2d_arr_to_c(lut))  # fills it
        return lut

    def get_hole_card_2_idx_lut(self):
        """
        获取实际手牌到索引的查找表
        
        返回:
            np.ndarray: 形状为[N_CARDS_IN_DECK, N_CARDS_IN_DECK]的数组
                - 每个元素表示对应手牌组合的索引
                - -2表示无效值
        """
        lut = np.full(shape=(HoldemRules.N_CARDS_IN_DECK, HoldemRules.N_CARDS_IN_DECK),
                      fill_value=-2, dtype=np.int16)
        self._clib.get_hole_card_2_idx_lut(self.np_2d_arr_to_c(lut))  # fills it
        return lut

    def get_idx_2_flop_lut(self):
        """
        获取翻牌索引到实际翻牌的查找表
        
        返回:
            np.ndarray: 形状为[n_boards_lut[FLOP], n_cards_out_lut[FLOP]]的数组
                - 每行表示一组翻牌的点数和花色
                - -2表示无效值
        """
        lut = np.full(shape=(
            self._n_boards_lut[Poker.FLOP],
            self._n_cards_out_lut[Poker.FLOP]),
            fill_value=-2, dtype=np.int8)
        self._clib.get_idx_2_flop_lut(self.np_2d_arr_to_c(lut))  # fills it
        return lut

    def get_idx_2_turn_lut(self):
        """
        获取转牌索引到实际转牌的查找表
        
        返回:
            np.ndarray: 形状为[n_boards_lut[TURN], n_cards_out_lut[TURN]]的数组
                - 每行表示一组转牌的点数和花色
                - -2表示无效值
        """
        lut = np.full(shape=(
            self._n_boards_lut[Poker.TURN],
            self._n_cards_out_lut[Poker.TURN]),
            fill_value=-2, dtype=np.int8)
        self._clib.get_idx_2_turn_lut(self.np_2d_arr_to_c(lut))  # fills it
        return lut

    def get_idx_2_river_lut(self):
        """
        获取河牌索引到实际河牌的查找表
        
        返回:
            np.ndarray: 形状为[n_boards_lut[RIVER], n_cards_out_lut[RIVER]]的数组
                - 每行表示一组河牌的点数和花色
                - -2表示无效值
        """
        lut = np.full(shape=(
            self._n_boards_lut[Poker.RIVER],
            self._n_cards_out_lut[Poker.RIVER]),
            fill_value=-2, dtype=np.int8)
        self._clib.get_idx_2_river_lut(self.np_2d_arr_to_c(lut))  # fills it
        return lut

    def get_1d_card(self, card_2d):
        """
        将二维卡牌表示转换为一维表示
        
        参数:
            card_2d (np.ndarray): 形状为[2]的数组，表示[点数,花色]
        
        返回:
            int8: 卡牌的一维表示（1-52）
        """
        return self._clib.get_1d_card(self.np_1d_arr_to_c(card_2d))

    def get_2d_card(self, card_1d):
        """
        将一维卡牌表示转换为二维表示
        
        参数:
            card_1d (int): 卡牌的一维表示（1-52）
        
        返回:
            np.ndarray: 形状为[2]的数组，表示[点数,花色]
        """
        card_2d = np.empty(shape=2, dtype=np.int8)
        self._clib.get_2d_card(card_1d, self.np_1d_arr_to_c(card_2d))
        return card_2d
