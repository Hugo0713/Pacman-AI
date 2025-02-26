# Pacman-AI

基于 UC Berkeley CS188 课程项目的 Pacman AI 实现，并添加 CUDA 加速的强化学习训练。

## 项目简介

本项目是对经典游戏 Pacman 的 AI 实现，包含多种算法实现以及 CUDA 加速的强化学习训练。这些项目不仅仅是构建游戏 AI，更重要的是教授 AI 的基础概念，如信息化状态空间搜索、概率推理和强化学习等。这些概念是自然语言处理、计算机视觉和机器人技术等实际应用领域的基础。

主要特点：

- 实现多种搜索算法 (DFS, BFS, UCS, A*)
- 多种 AI 代理实现（MinMax, AlphaBeta 剪枝等）
- 基于强化学习的 AI 训练
- CUDA 并行加速训练过程

## 环境要求

- Python 3.7+
- PyTorch
- CUDA Toolkit (用于 GPU 加速)
- NumPy

## 安装说明

1. 克隆仓库：

```bash
git clone https://github.com/hugo0713/Pacman-AI.git
cd Pacman-AI
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 运行游戏：

```bash
python pacman.py
```

## 项目结构与功能模块

### 1. 搜索 (Project 1)

实现了多种搜索算法来解决 Pacman 世界中的导航问题：

- 深度优先搜索 (DFS)
- 广度优先搜索 (BFS)
- 统一代价搜索 (UCS)
- A* 搜索

运行示例：

```bash
python pacman.py -l tinyMaze -p SearchAgent -a fn=dfs
python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs
python pacman.py -l bigMaze -p SearchAgent -a fn=astar
```

### 2. 博弈论 (Project 2)

将经典 Pacman 建模为对抗性和随机搜索问题：

- Minimax 算法
- Alpha-Beta 剪枝
- Expectimax 算法
- 评估函数设计

### 3. 逻辑推理 (Project 3)

使用布尔逻辑表示 Pacman 世界：

- 规划任务
- 定位
- 建图
- SLAM (同步定位与地图构建)

### 4. 贝叶斯网络与隐马尔可夫模型 (Project 4)

实现概率推理：

- 贝叶斯网络推理
- 前向算法
- 粒子采样
- 基于噪声距离读数的幽灵定位

### 5. 机器学习 (Project 5)

实现并应用多种机器学习模型：

- 感知器算法
- 神经网络
- 循环神经网络
- 数字分类
- 语言识别

### 6. 强化学习 (Project 6)

实现多种强化学习算法：

- 价值函数学习
- Q-learning
- 近似 Q-learning
- CUDA 加速训练

## CUDA 加速优化

本项目在原始 CS188 项目基础上添加了 CUDA 加速支持，主要包括以下内容：

### 1. 矩阵乘法加速

实现了一个基于 CUDA 的矩阵乘法（SGEMM）算法，使用共享内存来提高性能。该实现包括：

- **naiveSgemm**：在 CPU 上的简单矩阵乘法实现。
- **naiveSgemm2D**：在 GPU 上的简单矩阵乘法实现，使用共享内存来减少全局内存访问。
- **cuBLAS**：使用 NVIDIA 的 cuBLAS 库进行高效的矩阵乘法。

运行示例：

```bash
# 编译并运行矩阵乘法示例
make
./gemm
```

### 2. 向量加法加速

实现了一个基于 CUDA 的向量加法算法，包含：

- **vectorAddCPU**：在 CPU 上的简单向量加法实现。
- **vectorAddGPU**：在 GPU 上的向量加法实现，使用 CUDA 核函数进行并行计算。

运行示例：

```bash
# 编译并运行向量加法示例
make
./basic
```

### 3. 性能比较

在 README 中的示例代码中，分别测量了 CPU 和 GPU 的执行时间，并计算了加速比。通过使用 CUDA，能够显著提高计算性能，尤其是在处理大规模数据时。

### 4. 代码结构

CUDA 加速部分的代码位于 `cuda-playground-master/csrc/` 目录下，主要包括：

- `gemm.cu`：实现矩阵乘法的 CUDA 代码。
- `basic.cu`：实现向量加法的 CUDA 代码。
- `Makefile`：用于编译 CUDA 代码的 Makefile。

## 参考资料

本项目基于 UC Berkeley CS188 课程项目开发，原始项目由 John DeNero、Dan Klein、Pieter Abbeel 等人开发。

[CS188 项目主页](https://inst.eecs.berkeley.edu/~cs188/sp24/projects/)

## 许可证

本项目遵循 MIT 许可证开源。
