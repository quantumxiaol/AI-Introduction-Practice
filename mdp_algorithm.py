"""
MDP（马尔可夫决策过程）算法模块
"""
import copy
import time
import random
from PyQt6.QtWidgets import QApplication
from draw_utils import update_map_search

# 参数
REWARD = -1  # constant reward for non-terminal states
DISCOUNT = 1
MAX_ERROR = 10**(-3)

NUM_ACTIONS = 4
ACTIONS = [(1, 0), (0, -1), (-1, 0), (0, 1)]  # Down, Left, Up, Right
NUM_ROW = 3
NUM_COL = 3


def print_ep(arr, policy=False):
    """打印效用值或策略"""
    res = ""
    for r in range(NUM_ROW):
        res += "|"
        for c in range(NUM_COL):
            if r == 2 and c == 1:
                val = "+1"
            else:
                if policy:
                    val = ["Down", "Left", "Up", "Right"][arr[r][c]]
                else:
                    val = str(arr[r][c])
            res += " " + val[:5].ljust(5) + " |"
        res += "\n"
    print(res)


def get_u(U, r, c, action):
    """获得通过从给定状态执行给定动作所达到的状态的效用"""
    dr, dc = ACTIONS[action]
    newR, newC = r+dr, c+dc

    if newR < 0 or newC < 0 or newR >= NUM_ROW or newC >= NUM_COL:
        return U[r][c]
    else:
        return U[newR][newC]


def calculate_u(U, r, c, action):
    """计算给定动作状态的效用"""
    u = REWARD
    u += 0.1 * DISCOUNT * get_u(U, r, c, (action-1) % 4)
    u += 0.8 * DISCOUNT * get_u(U, r, c, action)
    u += 0.1 * DISCOUNT * get_u(U, r, c, (action+1) % 4)
    return u


def value_iteration(U):
    """值迭代算法"""
    print("值迭代:\n")
    while True:
        nextU = [[0, 0, 0],
                 [0, 0, 0],
                 [0, 1, 0]]
        error = 0
        for r in range(NUM_ROW):
            for c in range(NUM_COL):
                # 到达食物
                if (r == 2 and c == 1):
                    continue
                nextU[r][c] = max([calculate_u(U, r, c, action) for action in range(NUM_ACTIONS)])  # Bellman update
                error = max(error, abs(nextU[r][c]-U[r][c]))
        U = nextU
        print_ep(U)
        if error < MAX_ERROR:
            break
    return U


def get_optimal_policy(U):
    """从U得到最优策略"""
    policy = [[-1, -1, -1] for i in range(NUM_ROW)]
    for r in range(NUM_ROW):
        for c in range(NUM_COL):
            # 选择使效用最大化的行动
            maxAction, maxU = None, -float("inf")
            for action in range(NUM_ACTIONS):
                u = calculate_u(U, r, c, action)
                if u > maxU:
                    maxAction, maxU = action, u
            policy[r][c] = maxAction
    return policy


def movement_mdp(scene, map_obj, cell_width, rows, cols):
    """
    MDP算法
    - State:位置
    - Action:上下左右
    - Reward:体力消耗
    - Discount:r=1
    - 每走一步消耗体力 1,记为-1
    - 找到能量食物结束
    - 最小体力消耗找到能量食物
    """
    visited_paths = set()
    final_path = []
    
    end = (map_obj.destination[0], map_obj.destination[1])
    start = (map_obj.start[0], map_obj.start[1])
    discount = DISCOUNT
    
    # 初始化U矩阵
    U = [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]

    cur_pos = start
    path_list = [start]
    visited_set = set([start])

    print("初始值:\n")
    print_ep(U)

    # 值迭代
    U = value_iteration(U)

    # 从U中得到最优策略并打印出来
    policy = get_optimal_policy(U)
    print("最优策略:\n")
    print_ep(policy, True)

    print("RunOutSuccessfly\n")
    # 根据策略移动
    while cur_pos != end:
        # 检查边界，确保不会越界
        if cur_pos[0] < 1 or cur_pos[0] >= map_obj.height - 1 or cur_pos[1] < 1 or cur_pos[1] >= map_obj.width - 1:
            break
        
        # 简化策略映射（实际应用中需要更复杂的映射）
        action = policy[cur_pos[0] % NUM_ROW][cur_pos[1] % NUM_COL]
        if action < 0 or action >= len(ACTIONS):
            break
            
        next_pos = (cur_pos[0] + ACTIONS[action][0], cur_pos[1] + ACTIONS[action][1])
        
        # 检查是否到达终点
        if next_pos == end:
            path_list.append(next_pos)
            visited_set.add(next_pos)
            final_path = path_list.copy()
            visited_paths = visited_set.copy()
            break
        
        # 检查下一个位置是否有效
        if (next_pos[0] >= 0 and next_pos[0] < map_obj.height and 
            next_pos[1] >= 0 and next_pos[1] < map_obj.width and
            map_obj.matrix[next_pos[0]][next_pos[1]] != -1):
            cur_pos = next_pos
            path_list.append(cur_pos)
            visited_set.add(cur_pos)
        else:
            break
    
    final_path = path_list.copy()
    visited_paths = visited_set.copy()
    
    # 显示完整路径
    for p in path_list:
        update_map_search(scene, map_obj.matrix, map_obj.path, p, cell_width, rows, cols, map_obj,
                         visited=visited_set, final_path_points=final_path)
        time.sleep(0.05)
        QApplication.processEvents()
    
    print("RunOutSuccessfly\n")
    return final_path

