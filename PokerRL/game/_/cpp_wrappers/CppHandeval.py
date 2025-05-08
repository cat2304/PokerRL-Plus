# Copyright (c) 2019 Eric Steinberger


import ctypes
import os
from os.path import join as ospj

import numpy as np

from PokerRL._.CppWrapper import CppWrapper
from PokerRL.game._.rl_env.game_rules import HoldemRules


class CppHandeval(CppWrapper):
    """
    德州扑克手牌评估的C++库包装器
    
    这个类继承自CppWrapper，用于调用C++编写的高性能手牌评估库。
    主要功能包括：
    1. 评估单个手牌的强度
    2. 批量评估多个手牌组合的强度
    3. 支持52张牌的德州扑克规则
    
    使用示例：
    ```python
    evaluator = CppHandeval()
    # 评估单个手牌
    hand = np.array([[1,0], [2,0]], dtype=np.int8)  # 两张红桃A和红桃2
    board = np.array([[3,0], [4,0], [5,0], [6,0], [7,0]], dtype=np.int8)  # 红桃3-7
    rank = evaluator.get_hand_rank_52_holdem(hand, board)
    ```
    """

    def __init__(self):
        """
        初始化手牌评估器
        
        加载C++动态链接库并设置函数参数类型。
        根据操作系统自动选择正确的库文件扩展名(.so或.dll)
        """
        # 加载动态链接库
        super().__init__(path_to_dll=ospj(os.path.dirname(os.path.realpath(__file__)),
                                          "lib_hand_eval." + self.CPP_LIB_FILE_ENDING))
        
        # 设置单个手牌评估函数的参数类型
        self._clib.get_hand_rank_52_holdem.argtypes = [
            self.ARR_2D_ARG_TYPE,  # 手牌数组类型
            self.ARR_2D_ARG_TYPE   # 公共牌数组类型
        ]
        self._clib.get_hand_rank_52_holdem.restype = ctypes.c_int32  # 返回整数类型

        # 设置批量手牌评估函数的参数类型
        self._clib.get_hand_rank_all_hands_on_given_boards_52_holdem.argtypes = [
            self.ARR_2D_ARG_TYPE,  # 结果数组类型
            self.ARR_2D_ARG_TYPE,  # 公共牌数组类型
            ctypes.c_int32,        # 公共牌数量
            self.ARR_2D_ARG_TYPE,  # 手牌索引到实际手牌的查找表
            self.ARR_2D_ARG_TYPE   # 一维卡牌到二维卡牌的查找表
        ]
        self._clib.get_hand_rank_all_hands_on_given_boards_52_holdem.restype = None

    def get_hand_rank_52_holdem(self, hand_2d, board_2d):
        """
        评估单个手牌的强度
        
        参数:
            hand_2d (np.ndarray): 玩家手牌，形状为[5,2]的numpy数组
                - 每行表示一张牌 [点数,花色]
                - 点数范围: 1-13 (A=1, 2-10, J=11, Q=12, K=13)
                - 花色范围: 0-3 (0=红桃, 1=方块, 2=梅花, 3=黑桃)
            board_2d (np.ndarray): 公共牌，形状为[5,2]的numpy数组
                - 格式同手牌
                - 如果公共牌不足5张，用0填充

        返回:
            int: 表示最强5张牌组合的强度
                - 数值越大表示牌力越强
                - 用于比较不同手牌的相对大小
        """
        return self._clib.get_hand_rank_52_holdem(self.np_2d_arr_to_c(hand_2d), self.np_2d_arr_to_c(board_2d))

    def get_hand_rank_all_hands_on_given_boards_52_holdem(self, boards_1d, lut_holder):
        """
        批量评估多个手牌组合的强度
        
        参数:
            boards_1d (np.ndarray): 多个公共牌组合，形状为[N,5]的numpy数组
                - N是公共牌组合的数量
                - 每行表示一组公共牌
                - 每张牌用1-52的整数表示
            lut_holder: 查找表持有者，包含两个重要的查找表：
                - LUT_IDX_2_HOLE_CARDS: 将手牌索引映射到实际手牌
                - LUT_1DCARD_2_2DCARD: 将一维卡牌表示转换为二维表示

        返回:
            np.ndarray: 形状为[N, RANGE_SIZE]的数组
                - N是公共牌组合的数量
                - RANGE_SIZE是所有可能的手牌组合数量
                - 每个元素表示对应手牌组合的强度
                - -1表示该手牌组合在当前公共牌下不可用
        """
        # 验证输入数组的形状
        assert len(boards_1d.shape) == 2
        assert boards_1d.shape[1] == 5
        
        # 创建结果数组，初始值设为-1
        hand_ranks = np.full(shape=(boards_1d.shape[0], HoldemRules.RANGE_SIZE), fill_value=-1, dtype=np.int32)
        
        # 调用C++函数进行批量评估
        self._clib.get_hand_rank_all_hands_on_given_boards_52_holdem(
            self.np_2d_arr_to_c(hand_ranks),  # 结果数组
            self.np_2d_arr_to_c(boards_1d),   # 公共牌数组
            boards_1d.shape[0],               # 公共牌组合数量
            self.np_2d_arr_to_c(lut_holder.LUT_IDX_2_HOLE_CARDS),  # 手牌查找表
            self.np_2d_arr_to_c(lut_holder.LUT_1DCARD_2_2DCARD)    # 卡牌转换表
        )
        return hand_ranks
