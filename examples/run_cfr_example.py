# Copyright (c) 2019 Eric Steinberger


"""
本脚本在离散化无限注Leduc扑克的抽象版本中运行150次CFR迭代。
可用的动作包括：{弃牌(FOLD), 过牌/跟注(CHECK/CALL), 底池大小下注/加注(POT-SIZE-BET/RAISE)}
"""

from PokerRL.cfr.VanillaCFR import VanillaCFR
from PokerRL.game import bet_sets
from PokerRL.game.games import DiscretizedNLLeduc
from PokerRL.rl.base_cls.workers.ChiefBase import ChiefBase

if __name__ == '__main__':
    from PokerRL._.CrayonWrapper import CrayonWrapper

    # 设置迭代次数和实验名称
    n_iterations = 150
    name = "CFR_EXAMPLE"

    # 初始化ChiefBase用于记录
    # 对于ChiefBase来说，传递None给t_prof就足够了。我们只用它来记录；这个CFR实现不是分布式的。
    chief = ChiefBase(t_prof=None)
    
    # 初始化CrayonWrapper用于可视化
    crayon = CrayonWrapper(name=name,
                           path_log_storage=None,
                           chief_handle=chief,
                           runs_distributed=False,
                           runs_cluster=False,
                           )
    
    # 初始化VanillaCFR算法
    # 使用离散化无限注Leduc扑克游戏
    # 设置下注选项为仅底池大小
    cfr = VanillaCFR(name=name,
                     game_cls=DiscretizedNLLeduc,
                     agent_bet_set=bet_sets.POT_ONLY,
                     chief_handle=chief)

    # 运行CFR迭代
    for iter_id in range(n_iterations):
        print("迭代次数: ", iter_id)
        cfr.iteration()  # 执行一次CFR迭代
        crayon.update_from_log_buffer()  # 更新日志缓冲区
        crayon.export_all(iter_nr=iter_id)  # 导出所有数据
