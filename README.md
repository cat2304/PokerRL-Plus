## PokerRL

一个用于扑克游戏中多智能体深度强化学习的框架。

---

## 背景

关于不完全信息博弈的研究，长期以来主要集中于遍历整个游戏树的方法，例如以下论文：

* \[0] Regret Minimization（Zinkevich et al.）
* \[1] CFR+ 算法（Tammelin et al.）
* \[2] 蒙特卡洛采样式 CFR（Lanctot et al.）

最近涌现了结合深度（强化）学习的新算法，如：

* 神经虚拟自我对弈（NFSP）\[3]
* Regret Policy Gradients（RPG）\[4]
* 深度 CFR（Deep CFR）\[5]
* 单一深度 CFR（Single Deep CFR）\[8]

这些方法在只访问游戏状态子集的前提下，可以逼近纳什均衡。

---

## PokerRL 框架结构

### 算法组成部分

PokerRL 的算法框架如图所示：

* 绿色模块：训练用的 workers（例如 Learner、Rollout Worker）
* 蓝色模块：训练配置（TrainingProfile）
* 橙色模块：用于策略评估的 Evaluator Master 和 LBR Workers
* 黄色模块：评估方式，包括 AI 对 AI、AI 对人类

你的训练代理将被封装为 EvalAgent，用于游戏模拟、测试或比赛：

* `.../rl/base_cls/EvalAgentBase`
* 可用于 `.../game/AgentTournament` 或 `.../game/InteractiveGame`

几乎所有模块都可以使用少量代码（4 行以内）封装为分布式独立 Worker。

⚠️ 注：部分功能仅支持双人博弈。

---

## 策略评估方式

PokerRL 提供四种常用策略评估方式：

1. **Best Response（BR）**：

   * 精确计算可被利用度，仅适用于小型博弈。

2. **Local Best Response（LBR）**：

   * 计算 BR 的下界，可分布式运行。

3. **RL-BR（RL 版最优反应）**：

   * 用 Dueling DQN 训练反应策略，模拟 BR。

4. **Head-To-Head（H2H）**：

   * 同一个智能体以不同模式对战。

> 同时提供了 CFR、CFR+、Linear CFR 等基线算法实现。

---

## 性能与可扩展性

传统算法的性能瓶颈在于遍历整棵游戏树，而基于采样的 PokerRL 框架将主要计算负载转移至神经网络调用上。

PokerRL 提供了完整的深度学习 + 多智能体强化学习环境，支持通过 `ray` 实现：

* 本地多核并行训练
* 多机集群训练（Cloud or Cluster）

---

## 安装说明

### 本地安装

建议先安装：Anaconda/Miniconda 和 Docker。

```bash
conda create -n PokerAI python=3.6 -y
source activate PokerAI
pip install requests
conda install pytorch=0.4.1 -c pytorch
pip install PokerRL
```

如果你需要运行分布式：

```bash
pip install PokerRL[distributed]
```

### TensorBoard 监控（使用 PyCrayon）

```bash
docker run -d -p 8888:8888 -p 8889:8889 --name crayon alband/crayon
docker start crayon
```

浏览器打开：`http://localhost:8888`

### 测试是否安装成功

```bash
python -m unittest discover PokerRL/test
python examples/interactive_user_v_user.py  # 人类 vs 自己
python examples/run_cfrp_example.py         # 训练 CFR+
```

---

## 云端 & 分布式训练

PokerRL 封装了 `ray` 接口，支持以下两种部署方式：

1. **分布式（Distributed）**：单机多核并行训练
2. **集群（Cluster）**：跨多台机器并行训练

> 开关分布式/集群模式，只需修改 `TrainingProfile` 中的布尔值即可。

示例算法：

* [NFSP](https://github.com/TinkeringCode/Neural-Fictitous-Self-Play)
* [Single Deep CFR](https://github.com/TinkeringCode/Single-Deep-CFR)

---

## 小工具 & 可视化

PokerRL 支持可视化小型游戏树策略（PokerViz 工具）。

* Windows：解压 PokerViz 到 `C:/PokerViz`
* Linux：解压到 `~/PokerViz`

PokerRL 会自动检测该工具并在运行 BR 时导出可视化数据。

打开：

```
data/ ➜ data.js ➜ index.html
```

---

## 引用方式

```bibtex
@misc{steinberger2019pokerrl,
    author = {Eric Steinberger},
    title = {PokerRL},
    year = {2019},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/TinkeringCode/PokerRL}},
}
```

## 作者

* Eric Steinberger

## 协议

MIT License

## 鸣谢

* Alexander Mandt：帮助运行 ray 分布式集群
* Sebastian De Ro：提供 PokerViz 工具 & 批量扑克手牌评估器（C++）
