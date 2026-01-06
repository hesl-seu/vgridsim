# VGridSim: 面向车网互动的人工智能调控算法开源测试平台

**VGridSim** 是一个专为车网互动（Vehicle-to-Grid, V2G）场景设计的人工智能调控算法开源测试平台。

本平台旨在解决现有平台难以兼顾物理精度与计算扩展性的痛点。通过分层解耦架构，VGridSim 集成了高精度的配电网潮流计算与灵活的强化学习（RL）训练环境，支持从算法研发、训练到性能评估的全流程工作。

## 🌟 主要特性 (Features)

* **三层解耦架构**:
    * **系统场景层**: 物理设备建模与场景配置。
    * **计算控制层**: 两阶段潮流计算（Linear DistFlow + OpenDSS）与调度控制。
    * **接口评测层**: Gym 标准接口封装与可视化评估。
* **高精度两阶段潮流**: 结合线性化 `DistFlow`与 `OpenDSS`进行潮流计算，兼顾计算效率与物理精度。
* **丰富的设备建模**:
    * **V2G**: 支持多充电站、多桩并发、随机/指定行程的电动汽车充放电行为建模。
    * **分布式能源**: 光伏 (PV)、风电 (Wind) 及 储能系统 (ESS)。
    * **柔性互联**: 支持软开关 (SOP) 和常开点 (NOP) 的网络重构与功率控制。
* **全面的算法支持**:
    * **强化学习 (RL)**: 封装为标准 `Gymnasium` 环境，内置 `Stable-Baselines3` 支持 (PPO, SAC, DDPG, TD3)。
    * **最优基准 (Optimal Baseline)**: 基于 `Pyomo` 的混合整数规划模型，支持调用 Gurobi/Cplex 求解全局最优解作为性能锚点。
* **可视化交互**: 提供基于 PyQt 的图形化界面 (GUI)，支持场景参数配置、训练过程实时监控及结果的多维图表展示。

## 📂 项目结构 (Structure)

```text
VGridSim/
├── fpowerkit/                  # 内置的修改版电力系统建模工具库
├── custom_algorithms/          # 用户自定义算法存放目录
├── models/                     # 存放着已经训练好的RL模型
├── data/                       # 仿真数据
│   ├── grid_parameters.xlsx    # 配电网详细参数配置
│   └── my_ev_scenarios.csv     # 自定义电动汽车行程数据
├── results/                    # 仿真与训练结果输出目录
├── gui_main.py                 # 图形化界面启动入口
├── baseline.py                 # 离线最优基准算法 (Pyomo + Solver)
├── train_ppo.py                # PPO 算法训练脚本
├── train_sac.py                # SAC 算法训练脚本
├── train_ddpg.py                # DDPG 算法训练脚本
├── train_td3.py                # TD3 算法训练脚本
├── train_all_overnight_v2.py   # 批量训练脚本
├── two_stage_powerflow.py      # 两阶段潮流计算核心
├── grid_model.py               # 电网对象构建工厂
├── power_grid_env.py           # Gym 强化学习环境定义
└── requirements.txt            # 项目依赖列表

```

## 🛠️ 安装 (Installation)

本项目基于 Python 3.11 开发。建议使用 Conda 创建虚拟环境。

### 1. 克隆仓库

```bash
git clone [https://github.com/Gardener-0/vgridsim.git](https://github.com/hesl-seu/vgridsim.git)
cd vgridsim

```

### 2. 安装依赖

**注意**: 本项目依赖一个定制修改版的 `fpowerkit` 库，该库的源代码已包含在本项目根目录下的 `/fpowerkit` 文件夹中。**请勿通过 pip 安装 fpowerkit**，Python 运行时会自动加载项目内的版本。

请仅安装其他标准依赖：

```bash
pip install -r requirements.txt

```

*提示: 如果需要运行 `baseline.py` (最优基准)，请确保已安装 `Gurobi` (推荐) 或 `CBC` 等优化求解器。*


### 上述方法不能正常使用的话，建议直接下载压缩包

## 🚀 快速开始 (Quick Start)

### 方式一：使用图形化界面 (GUI)

这是最直观的使用方式，适合进行参数配置、场景预览和单次仿真演示，最推荐的方式。

```bash
python gui_main.py

```

*在界面中，你可以选择电网拓扑（IEEE 33/69/123）、设置EV规模、选择算法（RL或Optimal Baseline）并观察实时训练曲线与潮流结果。*

### 方式二：训练强化学习智能体

直接运行对应的训练脚本即可开始训练模型。训练日志和模型文件将保存在 `results/` 目录下。

```bash
# 训练 PPO 智能体
python train_ppo.py

# 训练 SAC 智能体
python train_sac.py

# 训练 DDPG 智能体
python train_ddpg.py

# 训练 TD3 智能体
python train_td3.py

```

### 方式三：运行最优基准 (Optimal Baseline)

计算当前场景下的理论全局最优解，用于评估 RL 算法的性能差距（Optimality Gap）。

```bash
python baseline.py

```

## 📊 评估与结果 (Evaluation)

平台自动计算并输出以下关键指标：

* **经济性**: 系统总运行成本、购电成本、网损成本。
* **安全性**: 节点电压分布、线路负载率、电压越限惩罚。
* **用户满意度**: 电动汽车充电满足率 (SoC 达标率)。

## 致谢

* **东南大学 胡秦然老师，钱涛老师**

如有问题，欢迎提交 Issue 。

```

```
