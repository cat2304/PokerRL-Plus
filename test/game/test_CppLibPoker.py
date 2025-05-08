# Copyright (c) 2019 Eric Steinberger


import unittest
from unittest import TestCase

import numpy as np

from PokerRL.game._.cpp_wrappers.CppHandeval import CppHandeval


class TestCppLib(TestCase):
    """测试C++实现的德州扑克牌型评估库"""

    def test_get_hand_rank_52_holdem(self):
        """测试52张牌的德州扑克牌型评估功能"""
        # 初始化C++实现的牌型评估器
        cpp_poker = CppHandeval()
        
        # 创建公共牌（5张）
        # 数组格式：[牌面值, 花色]
        # 这里创建的是：2♠, 2♦, J♥, 10♣, J♣
        b = np.array([[2, 0], [2, 3], [11, 1], [10, 2], [11, 2]], dtype=np.int8)
        
        # 创建手牌（2张）
        # 这里创建的是：J♦, 5♥
        h = np.array([[11, 3], [5, 1]], dtype=np.int8)
        
        # 测试牌型评估函数
        # 验证返回的是整数类型的牌型等级
        assert isinstance(cpp_poker.get_hand_rank_52_holdem(hand_2d=h, board_2d=b), int)


if __name__ == '__main__':
    unittest.main()
