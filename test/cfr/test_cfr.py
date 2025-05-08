# Copyright (c) 2019 Eric Steinberger


import unittest
from unittest import TestCase

from PokerRL.cfr.CFRPlus import CFRPlus
from PokerRL.cfr.LinearCFR import LinearCFR
from PokerRL.cfr.VanillaCFR import VanillaCFR
from PokerRL.game import bet_sets
from PokerRL.game.games import DiscretizedNLLeduc
from PokerRL.rl.base_cls.workers.ChiefBase import ChiefBase


class TestCFR(TestCase):
    """测试三种CFR算法的基本功能"""

    def test_run_CFR(self):
        """测试基础CFR算法"""
        n_iterations = 3  # 迭代次数
        name = "TESTING_CFR"

        # 创建Chief控制器和CFR实例
        chief = ChiefBase(t_prof=None)
        cfr = VanillaCFR(name=name,
                         game_cls=DiscretizedNLLeduc,  # 使用离散化的Leduc扑克游戏
                         agent_bet_set=bet_sets.POT_ONLY,  # 只允许下注底池大小
                         chief_handle=chief)

        cfr.reset()  # 重置CFR状态

        # 运行指定次数的迭代
        for iter_id in range(n_iterations):
            cfr.iteration()

    def test_run_CFRplus(self):
        """测试CFR+算法（CFR的改进版本）"""
        n_iterations = 3
        name = "TESTING_CFRplus"

        chief = ChiefBase(t_prof=None)
        cfr = CFRPlus(name=name,
                      game_cls=DiscretizedNLLeduc,
                      delay=0,  # 延迟参数设为0
                      agent_bet_set=bet_sets.POT_ONLY,
                      chief_handle=chief)

        cfr.reset()

        for iter_id in range(n_iterations):
            cfr.iteration()

    def test_run_linearCFR(self):
        """测试Linear CFR算法（线性版本的CFR）"""
        n_iterations = 3
        name = "TESTING_LinearCFR"

        chief = ChiefBase(t_prof=None)
        cfr = LinearCFR(name=name,
                        game_cls=DiscretizedNLLeduc,
                        agent_bet_set=bet_sets.POT_ONLY,
                        chief_handle=chief)

        cfr.reset()

        for iter_id in range(n_iterations):
            cfr.iteration()


if __name__ == '__main__':
    unittest.main()
