# Copyright (c) 2019 Eric Steinberger


"""
This script runs 150 iterations of CFR+ in a Leduc poker game with actions {FOLD, CHECK/CALL, POT-SIZE-BET/RAISE}.
It will store logs and tree files on your C: drive.
"""

from PokerRL.cfr.CFRPlus import CFRPlus
from PokerRL.game import bet_sets
from PokerRL.game.games import DiscretizedNLLeduc
from PokerRL.rl.base_cls.workers.ChiefBase import ChiefBase

if __name__ == '__main__':
    from PokerRL._.CrayonWrapper import CrayonWrapper

    n_iterations = 150
    name = "CFRplus_EXAMPLE"

    # 对于ChiefBase来说，传递None给t_prof就足够了。我们只用它来记录；这个CFR实现不是分布式的。
    chief = ChiefBase(t_prof=None)
    crayon = CrayonWrapper(name=name,
                           path_log_storage=None,
                           chief_handle=chief,
                           runs_distributed=False,
                           runs_cluster=False)
    cfr = CFRPlus(name=name,
                  game_cls=DiscretizedNLLeduc,
                  delay=0,
                  agent_bet_set=bet_sets.POT_ONLY,
                  chief_handle=chief)

    for iter_id in range(n_iterations):
        print("Iteration: ", iter_id)
        cfr.iteration()
        crayon.update_from_log_buffer()
        crayon.export_all(iter_nr=iter_id)
