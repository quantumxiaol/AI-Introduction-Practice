"""
搜索算法模块 - 提供各种路径搜索算法
"""
import copy
import time
from collections import deque
import queue
from PyQt6.QtWidgets import QApplication
from draw_utils import update_map_search


def get_successors(cur_pos, map_obj):
    """获取当前位置的所有后继节点"""
    able = []
    row, col = cur_pos[0], cur_pos[1]
    # 上方 - 检查边界
    if row > 0 and (map_obj.matrix[row-1][col] == 0 or map_obj.matrix[row-1][col] == 2):
        able.append(((row-1, col), [-1, 0], 1))
    # 下方 - 检查边界
    if row < map_obj.height - 1 and (map_obj.matrix[row+1][col] == 0 or map_obj.matrix[row+1][col] == 2):
        able.append(((row+1, col), [1, 0], 1))
    # 左方 - 检查边界
    if col > 0 and (map_obj.matrix[row][col-1] == 0 or map_obj.matrix[row][col-1] == 2):
        able.append(((row, col-1), [0, -1], 1))
    # 右方 - 检查边界
    if col < map_obj.width - 1 and (map_obj.matrix[row][col+1] == 0 or map_obj.matrix[row][col+1] == 2):
        able.append(((row, col+1), [0, 1], 1))
    return able


def movement_dfs(scene, map_obj, cell_width, rows, cols):
    """深度优先搜索算法"""
    visited_paths = set()
    final_path = []
    
    # 使用地图副本进行搜索，避免修改原始地图
    search_matrix = copy.deepcopy(map_obj.matrix)
    end = (map_obj.destination[0], map_obj.destination[1])
    start = (map_obj.start[0], map_obj.start[1])
    path_list = [start]
    visited_set = set()

    while path_list:
        cur_pos = path_list[-1]
        if cur_pos == end:
            print(path_list)
            print("RunOutSuccessfly\n")
            final_path = path_list.copy()
            break
        row, col = cur_pos
        # 已经走过
        search_matrix[row][col] = -2
        visited_set.add(cur_pos)
        # 上方 - 检查边界
        if row > 0 and (search_matrix[row-1][col] == 0 or search_matrix[row-1][col] == 2):
            path_list.append((row-1, col))
            search_matrix[row-1][col] = 1
            continue
        # 下方 - 检查边界
        elif row < map_obj.height - 1 and (search_matrix[row+1][col] == 0 or search_matrix[row+1][col] == 2):
            path_list.append((row+1, col))
            search_matrix[row+1][col] = 1
            continue
        # 左方 - 检查边界
        elif col > 0 and (search_matrix[row][col-1] == 0 or search_matrix[row][col-1] == 2):
            path_list.append((row, col-1))
            search_matrix[row][col-1] = 1
            continue
        # 右方 - 检查边界
        elif col < map_obj.width - 1 and (search_matrix[row][col+1] == 0 or search_matrix[row][col+1] == 2):
            path_list.append((row, col+1))
            search_matrix[row][col+1] = 1
            continue
        else:
            path_list.pop()
    else:
        print("Error\n")
        return None
        
    # 显示完整路径
    for p in path_list:
        update_map_search(scene, map_obj.matrix, map_obj.path, p, cell_width, rows, cols, map_obj, 
                         visited=visited_set, final_path_points=final_path)
        time.sleep(0.05)
        QApplication.processEvents()
    
    return final_path


def movement_bfs(scene, map_obj, cell_width, rows, cols):
    """广度优先搜索算法"""
    visited_paths = set()
    final_path = []
    
    # 使用地图副本进行搜索，避免修改原始地图
    search_matrix = copy.deepcopy(map_obj.matrix)
    end = (map_obj.destination[0], map_obj.destination[1])
    start = (map_obj.start[0], map_obj.start[1])
    path_list = []
    visited_set = set()
    
    dirs = [
        lambda x, y: (x+1, y),
        lambda x, y: (x-1, y),
        lambda x, y: (x, y-1),
        lambda x, y: (x, y+1)
    ]
    
    # 创建队列 起点入队,起点没有上一节点所以这里的联系用-1表示
    queue_obj = deque()
    queue_obj.append((start[0], start[1], -1))
    
    while len(queue_obj) > 0:
        curnode = queue_obj.popleft()
        path_list.append(curnode)
        visited_set.add((curnode[0], curnode[1]))
        # 找到迷宫终点跳出循环
        if curnode[0] == end[0] and curnode[1] == end[1]:
            cur = path_list[-1]
            # 存放最终路径结果
            path_result = []
            while cur[2] != -1:  # 只有起点的第三个元素才是-1
                path_result.append((cur[0], cur[1]))  # 路径不用储存节点之间的联系
                cur = path_list[cur[2]]  # 找到上一节点
            path_result.reverse()
            print(path_result)
            print("RunOutSuccessfly\n")
            final_path = path_result.copy()
            # 显示完整路径
            for p in path_result:
                update_map_search(scene, map_obj.matrix, map_obj.path, p, cell_width, rows, cols, map_obj,
                                 visited=visited_set, final_path_points=final_path)
                time.sleep(0.05)
                QApplication.processEvents()
            return final_path
        
        # 未找到终点执行循环
        for dir_func in dirs:
            nextnode = dir_func(curnode[0], curnode[1])
            # 判断下一节点是否可通过
            if search_matrix[nextnode[0]][nextnode[1]] == 0 or search_matrix[nextnode[0]][nextnode[1]] == 2:
                # 队列元素与nextnode形式不同，队列中要加入节点间的联系
                queue_obj.append((nextnode[0], nextnode[1], path_list.index(curnode)))
                # 将循环过的节点标记为走过
                search_matrix[nextnode[0]][nextnode[1]] = -2
    else:
        print("Error")
        return None


def movement_ucs(scene, map_obj, cell_width, rows, cols):
    """一致代价搜索算法"""
    visited_paths = set()
    final_path = []
    
    end = (map_obj.destination[0], map_obj.destination[1])
    start = (map_obj.start[0], map_obj.start[1])
    cur_pos = start
    path = [start]
    # 初始化相关参数
    result = []
    explored = set()
    visited_set = set()
    frontier = queue.PriorityQueue()
    # 定义起始状态，其中包括开始的位置，对应的行动方案和行动代价
    start_state = ((map_obj.start[0], map_obj.start[1]), [], 0)
    # 把起始状态放进frontier队列中
    frontier.put(start_state, 0)
    # 构造循环，循环读取frontier中的状态，进行判定
    while not frontier.empty():
        # 获取当前节点的各项信息
        (node, move, cost) = frontier.get()
        visited_set.add(node)
        # 如果弹出的节点状态满足目标要求，停止循环
        if node == (end[0], end[1]):
            result = move
            break
        # 如果该节点不满足目标要求，判定其是否访问过
        if node not in explored:
            explored.add(node)
            # 遍历这个节点的子节点，更新frontier队列
            for child, direction, step in get_successors(node, map_obj):
                newMove = move + [direction]
                newCost = cost + step
                newNode = (child, newMove, newCost)
                frontier.put(newNode, newCost)
    
    # 返回计算结果，即一个行动方案
    if not result:
        print("Error: No path found\n")
        return None
    
    # 构建最终路径
    final_path_list = [(map_obj.start[0], map_obj.start[1])]
    temp_pos = (map_obj.start[0], map_obj.start[1])
    for p in result:
        temp_pos = (temp_pos[0]+p[0], temp_pos[1]+p[1])
        final_path_list.append(temp_pos)
    final_path = final_path_list.copy()
    
    # 显示完整路径
    for p in result:
        cur_pos = (cur_pos[0]+p[0], cur_pos[1]+p[1])
        path.append(cur_pos)
        update_map_search(scene, map_obj.matrix, map_obj.path, cur_pos, cell_width, rows, cols, map_obj,
                         visited=visited_set, final_path_points=final_path)
        time.sleep(0.05)
        QApplication.processEvents()
    print("RunOutSuccessfly\n")
    print(path)
    return result


def movement_astar(scene, map_obj, cell_width, rows, cols):
    """A*搜索算法"""
    visited_paths = set()
    final_path = []
    
    # 使用地图副本进行搜索，避免修改原始地图
    lab = copy.deepcopy(map_obj.matrix)
    end = (map_obj.destination[0], map_obj.destination[1])
    start = (map_obj.start[0], map_obj.start[1])
    (i_s, j_s) = start
    (i_e, j_e) = end

    width = len(lab[0])
    height = len(lab)

    heuristic = lambda i, j: abs(i_e - i) + abs(j_e - j)
    comp = lambda state: state[2] + state[3]  # get the total cost

    # small variation for easier code, state is (coord_tuple, previous, path_cost, heuristic_cost)
    fringe = [((i_s, j_s), list(), 0, heuristic(i_s, j_s))]
    visited = {}
    visited_set = set()

    max_iterations = height * width * 10  # 防止无限循环
    iteration_count = 0
    
    while fringe and iteration_count < max_iterations:
        iteration_count += 1
        # get first state (least cost)
        state = fringe.pop(0)
        # goal check
        (i, j) = state[0]
        
        # 如果已经访问过且代价更高，跳过
        if (i, j) in visited and visited[(i, j)] <= state[2]:
            continue
            
        visited_set.add((i, j))
        visited[(i, j)] = state[2]
        
        if (i, j) == end:
            path = [state[0]] + state[1]
            path.reverse()
            print(path)
            print("RunOutSuccessfly\n")
            final_path = path.copy()
            # 显示完整路径
            for p in path:
                update_map_search(scene, map_obj.matrix, map_obj.path, (p[0], p[1]), cell_width, rows, cols, map_obj,
                                 visited=visited_set, final_path_points=final_path)
                time.sleep(0.05)
                QApplication.processEvents()
            return path

        # explore neighbor
        neighbor = list()
        if i > 0 and lab[i-1][j] >= 0:
            neighbor.append((i-1, j))
        if i < height - 1 and lab[i+1][j] >= 0:
            neighbor.append((i+1, j))
        if j > 0 and lab[i][j-1] >= 0:
            neighbor.append((i, j-1))
        if j < width - 1 and lab[i][j+1] >= 0:
            neighbor.append((i, j+1))

        for n in neighbor:
            # 跳过已访问的节点（除非找到更优路径）
            if n in visited_set:
                continue
                
            next_cost = state[2] + 1
            # 如果已经访问过且代价更高或相等，跳过
            if n in visited and visited[n] <= next_cost:
                continue
                
            # 检查fringe中是否已有该节点，如果有且代价更高，移除它
            fringe = [s for s in fringe if s[0] != n or s[2] < next_cost]
            
            fringe.append((n, [state[0]] + state[1], next_cost, heuristic(n[0], n[1])))

        # resort the list (SHOULD use a priority queue here to avoid re-sorting all the time)
        fringe.sort(key=comp)
    
    # 如果没有找到路径
    print("Error: No path found\n")
    return None

