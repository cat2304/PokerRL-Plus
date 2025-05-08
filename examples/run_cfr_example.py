# Copyright (c) 2019 Eric Steinberger


"""
This script runs 150 iterations of CFR in an abstracted version of Discrete No-Limit Leduc Poker with actions
{FOLD, CHECK/CALL, POT-SIZE-BET/RAISE}.
"""

from PokerRL.cfr.VanillaCFR import VanillaCFR
from PokerRL.game import bet_sets
from PokerRL.game.games import DiscretizedNLLeduc
from PokerRL.rl.base_cls.workers.ChiefBase import ChiefBase

if __name__ == '__main__':
    from PokerRL._.CrayonWrapper import CrayonWrapper

    n_iterations = 150
    name = "CFR_EXAMPLE"

    # 对于ChiefBase来说，传递None给t_prof就足够了。我们只用它来记录；这个CFR实现不是分布式的。
    chief = ChiefBase(t_prof=None)
    crayon = CrayonWrapper(name=name,
                           path_log_storage=None,
                           chief_handle=chief,
                           runs_distributed=False,
                           runs_cluster=False,
                           )
    cfr = VanillaCFR(name=name,
                     game_cls=DiscretizedNLLeduc,
                     agent_bet_set=bet_sets.POT_ONLY,
                     chief_handle=chief)

    for iter_id in range(n_iterations):
        print("Iteration: ", iter_id)
        cfr.iteration()
        crayon.update_from_log_buffer()
        crayon.export_all(iter_nr=iter_id)
