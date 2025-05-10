# Copyright (c) 2019 Eric Steinberger


"""
本脚本将启动一个3人无限注德州扑克游戏，使用离散的下注大小，您可以自己与自己进行游戏。
"""

from PokerRL.game.InteractiveGame import InteractiveGame
from PokerRL.game.games import DiscretizedNLHoldem

if __name__ == '__main__':
    # 设置游戏类型为离散化无限注德州扑克
    game_cls = DiscretizedNLHoldem
    # 配置游戏参数
    args = game_cls.ARGS_CLS(n_seats=3,  # 3个座位
                             bet_sizes_list_as_frac_of_pot=[
                                 0.2,    # 下注底池的20%
                                 0.5,    # 下注底池的50%
                                 1.0,    # 下注底池的100%
                                 2.0,    # 下注底池的200%
                                 1000.0  # 注意1000倍的底池将始终大于底池，因此代表全押
                             ],
                             stack_randomization_range=(0, 0,),  # 筹码随机化范围
                             )

    # 初始化交互式游戏
    # seats_human_plays_list=[0, 1, 2] 表示所有三个座位都由人类玩家控制
    game = InteractiveGame(env_cls=game_cls,
                           env_args=args,
                           seats_human_plays_list=[0, 1, 2],
                           )

    # 开始游戏
    game.start_to_play()
