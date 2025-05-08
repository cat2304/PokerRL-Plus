# Copyright (c) 2019 Eric Steinberger


import unittest
from unittest import TestCase

from PokerRL.game._.rl_env.base._Deck import DeckOfCards


class Test(TestCase):
    """测试扑克牌组（Deck）的基本功能"""

    def test_build(self):
        """测试牌组的初始化"""
        d = DeckOfCards()
        # 验证牌组大小是否正确
        # 牌组大小 = 牌面数 × 花色数
        assert d.deck_remaining.shape == (d.n_ranks * d.n_suits, 2)

    def test_draw(self):
        """测试抽牌功能"""
        # 进行20次测试
        for n in range(20):
            # 创建新的牌组
            deck = DeckOfCards()
            # 抽取n张牌
            cards = deck.draw(n)
            # 验证剩余牌数是否正确
            assert deck.deck_remaining.shape == (deck.n_ranks * deck.n_suits - n, 2)

            # 验证抽出的牌不会在剩余牌组中重复出现
            for card in cards:
                for _card in deck.deck_remaining:
                    assert not (card[0] == _card[0] and card[1] == _card[1])


if __name__ == '__main__':
    unittest.main()
