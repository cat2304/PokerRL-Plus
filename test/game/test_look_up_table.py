# Copyright (c) 2019 Eric Steinberger


import unittest
from unittest import TestCase

import numpy as np

from PokerRL.game.Poker import Poker
from PokerRL.game._.look_up_table import LutHolderHoldem, _LutGetterHoldem, _LutGetterLeduc
from PokerRL.game.games import StandardLeduc, DiscretizedNLHoldem


class TestLutGetterHoldem(TestCase):
    """测试德州扑克查找表生成器"""

    def test_get_1d_card_2_2d_card_lut(self):
        """测试一维牌到二维牌的映射表"""
        lg = _LutGetterHoldem(env_cls=DiscretizedNLHoldem)
        lut = lg.get_1d_card_2_2d_card_LUT()
        # 验证映射表大小：52张牌，每张牌2个属性（牌面值和花色）
        assert lut.shape == (52, 2)
        assert not np.any(lut == -2)  # 确保没有无效值

    def test_get_2d_card_2_1d_card_lut(self):
        """测试二维牌到一维牌的映射表"""
        lg = _LutGetterHoldem(env_cls=DiscretizedNLHoldem)
        lut = lg.get_2d_card_2_1d_card_LUT()
        # 验证映射表大小：13个牌面值，4种花色
        assert lut.shape == (13, 4)
        assert not np.any(lut == -2)

    def test_get_idx_2_hole_card_lut(self):
        """测试索引到手牌的映射表"""
        lg = _LutGetterHoldem(env_cls=DiscretizedNLHoldem)
        lut = lg.get_idx_2_hole_card_LUT()
        # 验证映射表大小：1326种可能的手牌组合，每手牌2张
        assert lut.shape == (1326, 2)
        assert not np.any(lut == -2)

    def test_get_hole_card_2_idx_lut(self):
        """测试手牌到索引的映射表"""
        lg = _LutGetterHoldem(env_cls=DiscretizedNLHoldem)
        lut = lg.get_hole_card_2_idx_LUT()
        # 验证映射表大小：52x52，表示任意两张牌的组合
        assert lut.shape == (52, 52)
        # 验证对称性：只使用上三角矩阵（避免重复计算）
        for i in range(52):
            for i2 in range(i + 1, 52):
                assert lut[i, i2] != -2
            for _i2 in range(0, i):
                assert lut[i, _i2] == -2

    def test_get_lut_card_in_what_range_idxs(self):
        """测试牌在范围索引中的位置查找表"""
        lg = _LutGetterHoldem(env_cls=DiscretizedNLHoldem)
        lut = lg.get_card_in_what_range_idxs_LUT()
        # 验证映射表大小：52张牌，每张牌可能出现在51个位置
        assert lut.shape == (52, 51)
        assert not np.any(lut == -2)

        # 验证每张牌在范围中出现的次数
        counts = np.zeros(1326, np.int32)
        for c in range(52):
            for h in lut[c]:
                counts[h] += 1
        assert np.all(counts == 2)  # 每张牌应该恰好出现两次


class TestLutGetterLeduc(TestCase):
    """测试Leduc扑克查找表生成器（简化版德州扑克）"""

    def test_get_1d_card_2_2d_card_lut(self):
        """测试一维牌到二维牌的映射表"""
        lg = _LutGetterLeduc(env_cls=StandardLeduc)
        lut = lg.get_1d_card_2_2d_card_LUT()
        # Leduc只有6张牌
        assert lut.shape == (6, 2)
        assert not np.any(lut == -2)

    def test_get_2d_card_2_1d_card_lut(self):
        """测试二维牌到一维牌的映射表"""
        lg = _LutGetterLeduc(env_cls=StandardLeduc)
        lut = lg.get_2d_card_2_1d_card_LUT()
        # Leduc有3个牌面值，2种花色
        assert lut.shape == (3, 2)
        assert not np.any(lut == -2)

    def test_get_idx_2_hole_card_lut(self):
        """测试索引到手牌的映射表"""
        lg = _LutGetterLeduc(env_cls=StandardLeduc)
        lut = lg.get_idx_2_hole_card_LUT()
        # Leduc只有6种可能的手牌组合
        assert lut.shape == (6, 1)
        assert not np.any(lut == -2)

    def test_get_hole_card_2_idx_lut(self):
        """测试手牌到索引的映射表"""
        lg = _LutGetterLeduc(env_cls=StandardLeduc)
        lut = lg.get_hole_card_2_idx_LUT()
        # Leduc的映射表大小
        assert lut.shape == (6, 1)
        assert not np.any(lut == -2)

    def test_get_lut_card_in_what_range_idxs(self):
        """测试牌在范围索引中的位置查找表"""
        lg = _LutGetterLeduc(env_cls=StandardLeduc)
        lut = lg.get_card_in_what_range_idxs_LUT()
        assert lut.shape == (6, 1)
        assert not np.any(lut == -2)

        # 验证每张牌在范围中出现的次数
        counts = np.zeros(6, np.int32)
        for c in range(6):
            for h in lut[c]:
                counts[h] += 1
        assert np.all(counts == 1)  # Leduc中每张牌只出现一次


class TestLutHolderHoldem(TestCase):
    """测试德州扑克查找表持有者"""

    def test_create(self):
        """测试查找表的创建和数据类型"""
        lh = DiscretizedNLHoldem.get_lut_holder()
        # 验证各种查找表的数据类型
        assert lh.LUT_1DCARD_2_2DCARD.dtype == np.dtype(np.int8)
        assert lh.LUT_2DCARD_2_1DCARD.dtype == np.dtype(np.int8)
        assert lh.LUT_IDX_2_HOLE_CARDS.dtype == np.dtype(np.int8)
        assert lh.LUT_HOLE_CARDS_2_IDX.dtype == np.dtype(np.int16)

    def test_get_1d_card(self):
        """测试一维牌表示方法"""
        lh = DiscretizedNLHoldem.get_lut_holder()
        # 测试各种牌面值的转换
        assert lh.get_1d_card(card_2d=[0, 3]) == 3
        assert lh.get_1d_card(card_2d=[12, 3]) == 51
        assert lh.get_1d_card(card_2d=np.array([0, 0], dtype=np.int8)) == 0
        assert lh.get_1d_card(card_2d=np.array([0, 0], dtype=np.int32)) == 0
        assert lh.get_1d_card(card_2d=Poker.CARD_NOT_DEALT_TOKEN_2D) == Poker.CARD_NOT_DEALT_TOKEN_1D

    def test_get_1d_cards(self):
        """测试多张牌的一维表示"""
        lh = DiscretizedNLHoldem.get_lut_holder()
        # 测试各种牌组合的转换
        assert np.array_equal(lh.get_1d_cards(cards_2d=np.array([[0, 3]])), np.array([3]))
        assert np.array_equal(lh.get_1d_cards(cards_2d=np.array([[0, 3], [12, 3]])), np.array([3, 51]))
        assert np.array_equal(lh.get_1d_cards(cards_2d=np.array([])), np.array([], np.int8))
        assert np.array_equal(lh.get_1d_cards(
            cards_2d=np.concatenate((np.array([[0, 0]]), Poker.CARD_NOT_DEALT_TOKEN_2D.reshape(-1, 2)))),
            np.array([0, Poker.CARD_NOT_DEALT_TOKEN_1D], dtype=np.int8))

    def test_get_2d_cards(self):
        """测试多张牌的二维表示"""
        lh = DiscretizedNLHoldem.get_lut_holder()
        # 测试各种牌组合的转换
        assert np.array_equal(lh.get_2d_cards(cards_1d=np.array([])), np.array([]))
        assert np.array_equal(lh.get_2d_cards(cards_1d=np.array([3])), np.array([[0, 3]]))
        assert np.array_equal(lh.get_2d_cards(cards_1d=np.array([3, 51])), np.array([[0, 3], [12, 3]]))
        assert np.array_equal(lh.get_2d_cards(cards_1d=np.array([])), np.array([]))
        assert np.array_equal(lh.get_2d_cards(cards_1d=np.array([0, Poker.CARD_NOT_DEALT_TOKEN_1D], dtype=np.int8)),
                              np.concatenate((np.array([[0, 0]]), Poker.CARD_NOT_DEALT_TOKEN_2D.reshape(-1, 2))))

    def test_get_range_idx_from_hole_cards(self):
        """测试从手牌获取范围索引"""
        lh = DiscretizedNLHoldem.get_lut_holder()
        n = 0
        # 验证所有可能的手牌组合的索引
        for c1 in range(51):
            for c2 in range(c1 + 1, 52):
                assert lh.LUT_HOLE_CARDS_2_IDX[c1, c2] == n
                n += 1

    def test_hole_card_luts(self):
        """测试手牌查找表的可逆性"""
        lh = DiscretizedNLHoldem.get_lut_holder()
        # 测试所有可能的手牌组合
        for h in range(1326):
            _c_1d = lh.get_1d_hole_cards_from_range_idx(h)  # 测试1d转换
            _c_2d = lh.get_2d_hole_cards_from_range_idx(h)  # 测试2d转换

            c_1d = (lh.LUT_IDX_2_HOLE_CARDS[h])
            c_2d = np.array([lh.LUT_1DCARD_2_2DCARD[c_1d[0]],
                             lh.LUT_1DCARD_2_2DCARD[c_1d[1]]],
                            dtype=np.int8)

            # 验证转换的一致性
            assert np.array_equal(c_1d, _c_1d)
            assert np.array_equal(c_2d, _c_2d)

        # 测试逆运算，验证1d和2d的映射都是唯一的
        for c1 in range(51):
            for c2 in range(c1 + 1, 50):
                cc1 = lh.LUT_1DCARD_2_2DCARD[c1]
                cc2 = lh.LUT_1DCARD_2_2DCARD[c2]
                hole_cards = np.array([cc1, cc2], dtype=np.int8)
                assert lh.get_range_idx_from_hole_cards(hole_cards) == \
                       lh.LUT_HOLE_CARDS_2_IDX[c1, c2]


if __name__ == '__main__':
    unittest.main()
