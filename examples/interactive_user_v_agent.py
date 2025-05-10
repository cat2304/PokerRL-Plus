# Copyright (c) 2019 Eric Steinberger

"""
交互式扑克游戏模板

本文件提供了一个模板，用于创建人类玩家与训练好的AI代理进行交互式扑克游戏。使用方法：

1. 将 "YourAlgorithmsEvalAgentCls" 替换为您具体算法的 EvalAgent 类（不是实例）
2. 更新 path_to_eval_agent 路径，指向您训练好的代理模型文件

注意：这是一个研究应用程序，其中：
- AI 无法看到人类玩家的牌
- 人类玩家可以看到 AI 的牌（用于研究目的）
- 游戏界面设计用于测试和评估，不用于竞技比赛

使用示例：
    eval_agent = DeepCFR.EvalAgent.load_from_disk(
        path_to_eval_agent="/path/to/your/trained/agent.pkl")
"""

from PokerRL.game.InteractiveGame import InteractiveGame

if __name__ == '__main__':
    # 加载训练好的代理
    eval_agent = YourAlgorithmsEvalAgentCls.load_from_disk(
        path_to_eval_agent="\\path\\to\\your\\eval_agent\\eval_agent.pkl")

    # 初始化交互式游戏
    # seats_human_plays_list=[0] 表示人类玩家将在座位0
    game = InteractiveGame(env_cls=eval_agent.env_bldr.env_cls,
                           env_args=eval_agent.env_bldr.env_args,
                           seats_human_plays_list=[0],
                           eval_agent=eval_agent,
                           )

    # 开始游戏
    game.start_to_play()
