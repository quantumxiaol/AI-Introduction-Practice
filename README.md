# AI-Introduction-Practice

人工智能导论实验课 - 吃豆人游戏实现

## 项目简介

本项目实现了两个吃豆人游戏实验：
1. **第一题**：使用多种搜索算法（DFS、BFS、UCS、A*）实现吃豆人自动寻路
2. **第二题**：使用值迭代（Value Iteration）方法实现吃豆人能量食物问题

项目采用 PyQt6 开发，支持多种地图生成算法和可视化搜索过程。

## 快速开始

### 环境配置

```bash
# 使用 uv 管理依赖
uv lock
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 运行程序

#### 第一题：搜索算法实现

```bash
# 运行吃豆人游戏（支持DFS、BFS、UCS、A*算法）
python pacman.py
```

**操作说明：**
- **WASD**：手动控制吃豆人移动
- **F1**：深度优先搜索（DFS）
- **F2**：广度优先搜索（BFS）
- **F3**：一致代价搜索（UCS）
- **F4**：A*搜索
- **F9**：重新开始游戏
- **菜单栏**：选择地图生成算法（预设地图、DFS、Prim、Kruskal、递归分割）

#### 第二题：值迭代方法

```bash
# 运行能量食物游戏（使用值迭代算法）
python pacmanenergyfood.py
```

**操作说明：**
- **WASD**：手动控制吃豆人移动
- **F1**：A*搜索算法
- **F2**：值迭代搜索（MDP）
- **F9**：重新开始游戏
- **菜单栏**：选择地图生成算法

## 项目结构

```
AI-Introduction-Practice/
├── pacman.py              # 第一题主程序（搜索算法）
├── pacmanenergyfood.py    # 第二题主程序（值迭代）
├── map_utils.py           # 地图工具类（地图生成和管理）
├── draw_utils.py          # 绘制工具函数（图形渲染）
├── search_algorithms.py   # 搜索算法实现（DFS、BFS、UCS、A*）
├── mdp_algorithm.py       # MDP算法实现（值迭代）
├── Maze.py                # 迷宫游戏（辅助模块）
├── mazeGenerator.py       # 迷宫生成器（辅助模块）
└── README.md              # 项目说明文档
```

## 算法实现

### 第一题：搜索算法

#### 1. 深度优先搜索（DFS）

**伪代码：**
```
function DFS(start, goal):
    stack = [start]
    visited = set()
    
    while stack is not empty:
        current = stack.pop()
        
        if current == goal:
            return path
        
        visited.add(current)
        
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                stack.push(neighbor)
    
    return None  // 未找到路径
```

**Python实现：** `search_algorithms.py::movement_dfs()`

使用栈（LIFO）实现，每次扩展最深的节点。特点：
- 使用 `path_list` 作为栈存储路径
- 通过 `visited_set` 记录已访问节点
- 回溯时使用 `path_list.pop()` 移除节点

#### 2. 广度优先搜索（BFS）

**伪代码：**
```
function BFS(start, goal):
    queue = [start]
    visited = set()
    parent = {}
    
    while queue is not empty:
        current = queue.dequeue()
        
        if current == goal:
            return reconstruct_path(parent, start, goal)
        
        visited.add(current)
        
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                queue.enqueue(neighbor)
                parent[neighbor] = current
    
    return None  // 未找到路径
```

**Python实现：** `search_algorithms.py::movement_bfs()`

使用队列（FIFO）实现，每次扩展最浅的节点。特点：
- 使用 `deque` 作为队列
- 通过 `path_list` 存储节点及其父节点索引
- 找到目标后通过父节点索引回溯路径

#### 3. 一致代价搜索（UCS）

**伪代码：**
```
function UCS(start, goal):
    priority_queue = PriorityQueue()
    priority_queue.put((0, start, []))
    visited = set()
    
    while priority_queue is not empty:
        cost, current, path = priority_queue.get()
        
        if current == goal:
            return path
        
        if current in visited:
            continue
        
        visited.add(current)
        
        for neighbor, action, step_cost in get_neighbors(current):
            new_cost = cost + step_cost
            new_path = path + [action]
            priority_queue.put((new_cost, neighbor, new_path))
    
    return None  // 未找到路径
```

**Python实现：** `search_algorithms.py::movement_ucs()`

使用优先队列实现，按累计代价排序。特点：
- 使用 `queue.PriorityQueue` 存储状态
- 状态格式：`(位置, 动作序列, 累计代价)`
- 每次选择代价最小的节点扩展

#### 4. A*搜索算法

**伪代码：**
```
function A_STAR(start, goal):
    open_set = PriorityQueue()
    open_set.put((h(start), start, [], 0, h(start)))
    closed_set = set()
    g_score = {start: 0}
    
    while open_set is not empty:
        f, current, path, g, h = open_set.get()
        
        if current == goal:
            return reconstruct_path(path)
        
        if current in closed_set:
            continue
        
        closed_set.add(current)
        
        for neighbor in get_neighbors(current):
            tentative_g = g + 1
            
            if neighbor in closed_set:
                continue
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                open_set.put((f_score, neighbor, path + [current], tentative_g, f_score - tentative_g))
    
    return None  // 未找到路径
```

**Python实现：** `search_algorithms.py::movement_astar()`

使用启发式函数优化搜索。特点：
- 评估函数：`f(n) = g(n) + h(n)`
  - `g(n)`：从起点到当前节点的实际代价
  - `h(n)`：从当前节点到目标的启发式估计（曼哈顿距离）
- 使用 `fringe` 列表存储状态，按 `f(n)` 排序
- 启发式函数：`h(i, j) = |i_e - i| + |j_e - j|`（曼哈顿距离）

### 第二题：值迭代方法（MDP）

#### 值迭代算法

**伪代码：**
```
function VALUE_ITERATION(states, actions, reward, discount, threshold):
    U = initialize_utilities(states)  // 初始化为0，目标状态为+1
    U' = copy(U)
    
    while true:
        U = U'
        delta = 0
        
        for each state s in states:
            if s is terminal:
                continue
            
            // Bellman更新
            U'[s] = max over actions a:
                sum over next_states s':
                    P(s'|s,a) * [R(s,a,s') + discount * U[s']]
            
            delta = max(delta, |U'[s] - U[s]|)
        
        if delta < threshold:
            break
    
    return U

function GET_OPTIMAL_POLICY(U, states, actions):
    policy = {}
    
    for each state s in states:
        best_action = None
        best_value = -infinity
        
        for each action a in actions:
            value = sum over next_states s':
                P(s'|s,a) * [R(s,a,s') + discount * U[s']]
            
            if value > best_value:
                best_value = value
                best_action = a
        
        policy[s] = best_action
    
    return policy
```

**Python实现：** `mdp_algorithm.py::value_iteration()` 和 `get_optimal_policy()`

**参数设置：**
- **状态空间**：3x3网格位置
- **动作空间**：上下左右4个方向
- **奖励函数**：每步-1，到达目标+100
- **折扣因子**：γ = 1
- **转移概率**：80%按预期方向移动，10%向左偏，10%向右偏
- **收敛条件**：最大误差 < 10⁻³

**Bellman更新公式：**
```
U(s) = max_a Σ_{s'} P(s'|s,a) * [R(s,a,s') + γ * U(s')]
```

其中：
- `P(s'|s,a)`：从状态s执行动作a到达状态s'的概率
- `R(s,a,s')`：即时奖励
- `γ`：折扣因子

## 地图生成算法

项目支持多种地图生成算法：

1. **预设地图**：使用固定的砖块地图（31x31或5x5）
2. **DFS算法**：深度优先搜索生成迷宫
3. **Prim算法**：Prim最小生成树算法
4. **Kruskal算法**：Kruskal最小生成树算法
5. **递归分割**：递归分割算法生成迷宫

实现位置：`map_utils.py::PacmanMap` 类

## 游戏特性

- ✅ 支持手动控制和自动搜索
- ✅ 可视化搜索过程（显示已访问节点和最终路径）
- ✅ 多种地图生成算法
- ✅ 实时显示搜索进度
- ✅ 支持地图切换和重新开始

## 任务要求

### 第一题（10分）

**题目说明：**

关于搜索方面：采用 Python3.x 开发语言，编写程序实现一个简单吃豆人（Pac-Man）小游戏，界面采用 tkinter 包。白色为食物，黄色为 pacman，蓝色为幽灵 ghost。Pacman 通过移动（上下左右，有墙壁阻挡不可移动）可以吃掉食物，pacman 遇到幽灵会被吃掉，幽灵不吃食物，幽灵移动（上下左右，有墙壁阻挡不可移动）随机。以下两个问题至少选择一个问题使用算法自动实现。

（1）使用深度优先、广度优先、一致代价和 A*四种搜索策略搜索目标；

（2）使用 min-max 策略，pac-man 吃掉所有食物而不被幽灵吃掉。

程序采用tkinter开发，地图大小为31×31。

**算法说明：**

- **深度优先（Depth-First Search）**：扩展最深的那个节点。使用LIFO栈。
- **广度优先（Breadth-First Search）**：扩展最浅的那个节点。使用FIFO队列。
- **一致代价（Uniform-Cost Search）**：扩展代价最小的那个节点。使用具有优先级的队列（优先级：累计代价/cumulative cost）
- **A***：UCS按照路径代价排序，即后向代价g(n) - Backward；贪心按照目标接近排序，即前向代价h(n) - Forward；A算法按照两者之和排序，即：f(n) = g(n) + h(n)

可相容的启发式方法是使用A*算法核心关键

**实现状态：** ✅ 已完成
- 实现了问题（1）：DFS、BFS、UCS、A*四种算法均已实现
- 支持可视化搜索过程
- 支持多种地图生成算法
- 注：本项目使用PyQt6替代tkinter实现图形界面

### 第二题（10分）

**题目说明：**

Pacman 吃能量食物，信息如下：
- State：位置
- Action：上下左右
- Reward：体力消耗
- Discount：r=1
- 每走一步消耗体力 1，记为-1
- 找到能量食物结束
- 最小体力消耗找到能量食物

以下两个问题至少选择一个问题使用算法自动实现。

（1）使用值迭代方法计算各个状态值；

（2）使用策略迭代方法计算最优行为策略。


## 参考资料

- A*搜索算法：使用曼哈顿距离作为启发式函数
- 值迭代算法：基于Bellman方程的最优值函数求解
- MDP理论：马尔可夫决策过程基础


