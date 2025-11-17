"""
地图工具类 - 提供地图管理和基础数据结构
"""
import numpy as np
import copy
import random

# 默认砖块地图（31x31）
BRICK_MATRIX_31 = [
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1, 0, 0, 0,-1, 0, 0,-1, 0, 0, 0,-1,-1,-1,-1, 0,-1, 0,-1,-1,-1, 0, 0, 0, 0, 0,-1, 0,-1,-1],
    [-1,-1, 0,-1, 0,-1, 0, 0,-1, 0,-1, 0, 0, 0, 0,-1, 0,-1, 0, 0, 0, 0, 0,-1,-1,-1, 0,-1, 0,-1,-1],
    [-1,-1, 0,-1, 0,-1, 0, 0,-1, 0,-1,-1,-1,-1, 0,-1, 0,-1, 0,-1,-1,-1, 0, 0, 0, 0, 0,-1, 0,-1,-1],
    [-1,-1, 0,-1, 0,-1, 0, 0, 0, 0,-1, 0,-1,-1, 0,-1, 0,-1, 0, 0, 0, 0, 0,-1, 0,-1, 0,-1, 0, 0,-1],
    [-1,-1, 0, 0, 0,-1, 0, 0,-1, 0,-1, 0,-1,-1, 0, 0, 0,-1, 0,-1,-1,-1, 0,-1, 0,-1, 0, 0, 0,-1,-1],
    [-1,-1, 0,-1, 0,-1, 0, 0,-1, 0,-1, 0, 0, 0, 0,-1, 0, 0, 0, 0,-1,-1, 0,-1, 0, 0,-1,-1,-1,-1,-1],
    [-1, 0, 0,-1,-1,-1,-1,-1,-1, 0,-1, 0,-1,-1, 0,-1,-1, 0,-1, 0, 0, 0, 0,-1,-1, 0, 0,-1,-1,-1,-1],
    [-1, 0,-1,-1, 0,-1, 0,-1,-1, 0,-1, 0,-1,-1, 0,-1,-1, 0,-1,-1,-1,-1, 0,-1, 0,-1,-1,-1,-1, 0,-1],
    [-1, 0,-1,-1, 0,-1, 0,-1, 0, 0, 0, 0,-1,-1, 0,-1,-1, 0,-1, 0,-1,-1, 0,-1, 0, 0, 0, 0,-1, 0,-1],
    [-1, 0, 0, 0, 0, 0, 0,-1, 0,-1, 0, 0,-1,-1, 0,-1,-1, 0,-1, 0,-1,-1, 0,-1, 0,-1,-1, 0,-1, 0,-1],
    [-1, 0,-1,-1, 0,-1, 0,-1, 0,-1, 0, 0,-1,-1, 0,-1,-1, 0,-1, 0,-1,-1, 0,-1, 0,-1,-1, 0,-1, 0,-1],
    [-1, 0, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0,-1,-1, 0, 0, 0,-1,-1, 0,-1, 0, 0, 0, 0,-1, 0,-1],
    [-1, 0,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1,-1, 0,-1, 0,-1,-1, 0,-1, 0,-1,-1, 0,-1, 0,-1],
    [-1, 0,-1,-1, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0,-1, 0,-1,-1, 0,-1, 0,-1,-1, 0,-1, 0,-1],
    [-1, 0,-1,-1, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0,-1,-1, 0, 0, 0, 0,-1, 0,-1, 0,-1],
    [-1, 0,-1,-1, 0,-1, 0,-1, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1,-1,-1,-1, 0, 0, 0,-1],
    [-1, 0,-1,-1, 0,-1, 0,-1, 0,-1,-1,-1, 0,-1, 0,-1, 0,-1,-1, 0,-1,-1,-1,-1,-1,-1,-1,-1, 0,-1,-1],
    [-1, 0, 0,-1,-1,-1, 0, 0, 0,-1,-1,-1, 0,-1, 0,-1, 0, 0,-1, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0,-1,-1],
    [-1, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0,-1, 0,-1,-1, 0,-1, 0, 0, 0, 0, 0,-1,-1, 0,-1,-1,-1,-1],
    [-1,-1, 0,-1, 0, 0, 0,-1,-1,-1,-1,-1, 0,-1, 0,-1,-1, 0,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0,-1],
    [-1,-1, 0,-1,-1,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0, 0, 0, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0,-1],
    [-1,-1, 0, 0, 0,-1, 0,-1, 0, 0, 0,-1, 0,-1, 0,-1,-1, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1],
    [-1,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0, 0, 0,-1,-1, 0,-1,-1,-1, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0, 0,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1],
    [-1,-1, 0, 0, 0,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0, 0, 0, 0,-1,-1,-1,-1,-1,-1,-1, 0,-1],
    [-1,-1, 0, 0, 0, 0, 0,-1, 0,-1, 0, 0, 0,-1,-1,-1, 0,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0,-1],
    [-1,-1, 0,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0,-1,-1,-1,-1, 0,-1,-1],
    [-1,-1, 0, 0, 0, 0, 0,-1,-1,-1,-1, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1, 0,-1,-1,-1, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
]

# 默认砖块地图（5x5）
BRICK_MATRIX_5 = [
    [-1,-1,-1,-1,-1],
    [-1, 0, 0, 0,-1],
    [-1, 0, 0, 0,-1],
    [-1, 0, 0, 0,-1],
    [-1,-1,-1,-1,-1]
]


class UnionSet(object):
    """并查集实现"""
    def __init__(self, arr):
        self.parent = {pos: pos for pos in arr}
        self.count = len(arr)

    def find(self, root):
        if root == self.parent[root]:
            return root
        return self.find(self.parent[root])

    def union(self, root1, root2):
        self.parent[self.find(root1)] = self.find(root2)


class PacmanMap:
    """吃豆人地图类"""
    def __init__(self, width=31, height=31, start=None, destination=None):
        assert width >= 5 and height >= 5, "Length of width or height must be larger than 5."
        self.width = (width // 2) * 2 + 1
        self.height = (height // 2) * 2 + 1
        
        if start is None:
            if height == 5:
                self.start = [1, 1]
            else:
                self.start = [14, 15]
        else:
            self.start = start
            
        if destination is None:
            if height == 5:
                self.destination = [self.height - 2, self.width - 3]
            else:
                self.destination = [self.height - 2, self.width - 2]
        else:
            self.destination = destination
            
        self.matrix = None
        self.path = []

    def print_matrix(self):
        """打印地图矩阵"""
        matrix = copy.deepcopy(self.matrix)
        for p in self.path:
            matrix[p[0]][p[1]] = 1
        for i in range(self.height):
            for j in range(self.width):
                if matrix[i][j] == -1:
                    print('B', end='')
                elif matrix[i][j] == 0:
                    print('  ', end='')
                elif matrix[i][j] == 1:
                    print('M', end='')
                elif matrix[i][j] == 2:
                    print('F', end='')
                elif matrix[i][j] == 3:
                    print('G', end='')
            print('')

    def generate_matrix(self, mode='brick', new_matrix=None):
        """
        生成地图矩阵
        mode: 'brick' - 预设砖块地图, 'dfs' - DFS算法, 'prim' - Prim算法, 
              'kruskal' - Kruskal算法, 'split' - 递归分割算法
        """
        if mode == 'brick':
            self.generate_matrix_brick()
        elif mode == 'dfs':
            self.generate_matrix_dfs()
        elif mode == 'prim':
            self.generate_matrix_prim()
        elif mode == 'kruskal':
            self.generate_matrix_kruskal()
        elif mode == 'split':
            self.generate_matrix_split()
        else:
            self.generate_matrix_brick()
        
        # 确保起点和终点正确设置
        self.matrix[self.start[0]][self.start[1]] = 1
        self.matrix[self.destination[0]][self.destination[1]] = 2

    def generate_matrix_brick(self):
        """使用预设砖块地图生成"""
        if self.height == 5:
            self.matrix = copy.copy(BRICK_MATRIX_5)
        else:
            self.matrix = copy.copy(BRICK_MATRIX_31)

    def generate_matrix_dfs(self):
        """使用DFS算法生成迷宫地图"""
        self.matrix = -np.ones((self.height, self.width))
        self.matrix[self.start[0], self.start[1]] = 0
        self.matrix[self.destination[0], self.destination[1]] = 0

        visit_flag = [[0 for i in range(self.width)] for j in range(self.height)]

        def check(row, col, row_, col_):
            temp_sum = 0
            for d in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                temp_sum += self.matrix[row_ + d[0]][col_ + d[1]]
            return temp_sum <= -3

        def dfs(row, col):
            visit_flag[row][col] = 1
            self.matrix[row][col] = 0
            # 对于小地图，使用不同的终止条件
            if self.height <= 5 or self.width <= 5:
                # 小地图：如果已经到达起点附近就返回
                if abs(row - self.start[0]) <= 1 and abs(col - self.start[1]) <= 1:
                    return
            else:
                if row == self.start[0] and col == self.start[1] + 1:
                    return

            directions = [[0, 2], [0, -2], [2, 0], [-2, 0]]
            random.shuffle(directions)
            for d in directions:
                row_, col_ = row + d[0], col + d[1]
                if row_ > 0 and row_ < self.height - 1 and col_ > 0 and col_ < self.width - 1 and visit_flag[row_][col_] == 0 and check(row, col, row_, col_):
                    if row == row_:
                        mid_col = min(col, col_) + 1
                        if 0 <= mid_col < self.width:
                            visit_flag[row][mid_col] = 1
                            self.matrix[row][mid_col] = 0
                    else:
                        mid_row = min(row, row_) + 1
                        if 0 <= mid_row < self.height:
                            visit_flag[mid_row][col] = 1
                            self.matrix[mid_row][col] = 0
                    dfs(row_, col_)

        # 对于小地图，从终点直接向起点生成路径
        if self.height <= 5 or self.width <= 5:
            # 简单连通起点和终点
            cur_row, cur_col = self.destination[0], self.destination[1]
            while cur_row != self.start[0] or cur_col != self.start[1]:
                if cur_row > self.start[0]:
                    cur_row -= 1
                elif cur_row < self.start[0]:
                    cur_row += 1
                elif cur_col > self.start[1]:
                    cur_col -= 1
                elif cur_col < self.start[1]:
                    cur_col += 1
                if 0 <= cur_row < self.height and 0 <= cur_col < self.width:
                    self.matrix[cur_row][cur_col] = 0
        else:
            dfs(self.destination[0], self.destination[1] - 1)
            if 0 <= self.start[1] + 1 < self.width:
                self.matrix[self.start[0], self.start[1] + 1] = 0
        
        # 确保起点和终点周围有通路
        for d in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
            r, c = self.start[0] + d[0], self.start[1] + d[1]
            if 0 <= r < self.height and 0 <= c < self.width:
                if self.matrix[r][c] == -1:
                    self.matrix[r][c] = 0
            r, c = self.destination[0] + d[0], self.destination[1] + d[1]
            if 0 <= r < self.height and 0 <= c < self.width:
                if self.matrix[r][c] == -1:
                    self.matrix[r][c] = 0

    def generate_matrix_prim(self):
        """使用Prim算法生成迷宫地图"""
        # 对于太小的地图，使用DFS算法代替
        if self.height < 7 or self.width < 7:
            self.generate_matrix_dfs()
            return
            
        self.matrix = -np.ones((self.height, self.width))

        def check(row, col):
            temp_sum = 0
            for d in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                temp_sum += self.matrix[row + d[0]][col + d[1]]
            return temp_sum < -3
            
        queue = []
        # 确保随机选择的起始点在有效范围内
        if self.height > 3 and self.width > 3:
            row = (np.random.randint(1, self.height - 1) // 2) * 2 + 1
            col = (np.random.randint(1, self.width - 1) // 2) * 2 + 1
        else:
            row, col = 1, 1
        queue.append((row, col, -1, -1))
        
        max_iterations = (self.height * self.width) // 2
        iteration_count = 0
        
        while len(queue) != 0 and iteration_count < max_iterations:
            iteration_count += 1
            if len(queue) == 0:
                break
            idx = np.random.randint(0, len(queue))
            row, col, r_, c_ = queue.pop(idx)
            if check(row, col):
                self.matrix[row, col] = 0
                if r_ != -1 and row == r_:
                    self.matrix[row][min(col, c_) + 1] = 0
                elif r_ != -1 and col == c_:
                    self.matrix[min(row, r_) + 1][col] = 0
                for d in [[0, 2], [0, -2], [2, 0], [-2, 0]]:
                    row_, col_ = row + d[0], col + d[1]
                    if row_ > 0 and row_ < self.height - 1 and col_ > 0 and col_ < self.width - 1 and self.matrix[row_][col_] == -1:
                        queue.append((row_, col_, row, col))

        # 确保起点和终点与迷宫内部连通
        self.matrix[self.start[0], self.start[1]] = 0
        self.matrix[self.destination[0], self.destination[1]] = 0
        # 确保起点和终点周围有通路
        for d in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
            r, c = self.start[0] + d[0], self.start[1] + d[1]
            if 0 <= r < self.height and 0 <= c < self.width and self.matrix[r][c] == -1:
                self.matrix[r][c] = 0
            r, c = self.destination[0] + d[0], self.destination[1] + d[1]
            if 0 <= r < self.height and 0 <= c < self.width and self.matrix[r][c] == -1:
                self.matrix[r][c] = 0

    def generate_matrix_split(self):
        """使用递归分割算法生成迷宫地图"""
        # 对于太小的地图（小于7x7），使用DFS算法代替
        if self.height < 7 or self.width < 7:
            self.generate_matrix_dfs()
            return
            
        self.matrix = -np.zeros((self.height, self.width))
        self.matrix[0, :] = -1
        self.matrix[self.height - 1, :] = -1
        self.matrix[:, 0] = -1
        self.matrix[:, self.width - 1] = -1

        # 随机生成位于(start, end)之间的偶数
        def get_random(start, end):
            if start >= end:
                return start
            # 确保范围有效
            if end - start <= 1:
                return start
            # 尝试多次找到偶数
            for _ in range(10):  # 最多尝试10次
                rand = np.random.randint(start, end)
                if rand & 0x1 == 0:
                    return rand
            # 如果找不到偶数，返回最接近的偶数
            if start & 0x1 == 0:
                return start
            return start + 1 if start + 1 < end else start

        # split函数的四个参数分别是左上角的行数、列数，右下角的行数、列数，墙壁只能在偶数行，偶数列
        def split(lr, lc, rr, rc):
            if rr - lr < 2 or rc - lc < 2:
                return

            # 生成墙壁,墙壁只能是偶数点
            cur_row, cur_col = get_random(lr, rr), get_random(lc, rc)
            for i in range(lc, rc + 1):
                self.matrix[cur_row][i] = -1
            for i in range(lr, rr + 1):
                self.matrix[i][cur_col] = -1
            
            # 挖穿三面墙得到连通图，挖孔的点只能是偶数点
            wall_list = [
                ("left", cur_row, [lc + 1, cur_col - 1]),
                ("right", cur_row, [cur_col + 1, rc - 1]), 
                ("top", cur_col, [lr + 1, cur_row - 1]),
                ("down", cur_col, [cur_row + 1, rr - 1])
            ]
            random.shuffle(wall_list)
            opened_walls = 0
            for wall in wall_list:
                if wall[2][1] - wall[2][0] < 0:
                    continue
                # 确保范围有效
                if wall[2][0] > wall[2][1]:
                    continue
                    
                try:
                    if wall[0] in ["left", "right"]:
                        hole_pos = get_random(wall[2][0], wall[2][1] + 1)
                        if hole_pos + 1 < self.width:
                            self.matrix[wall[1], hole_pos + 1] = 0
                            opened_walls += 1
                    else:
                        hole_pos = get_random(wall[2][0], wall[2][1] + 1)
                        if hole_pos + 1 < self.height:
                            self.matrix[hole_pos + 1, wall[1]] = 0
                            opened_walls += 1
                except:
                    pass
                    
                # 至少打开3面墙
                if opened_walls >= 3:
                    break

            # 递归
            split(lr + 2, lc + 2, cur_row - 2, cur_col - 2)
            split(lr + 2, cur_col + 2, cur_row - 2, rc - 2)
            split(cur_row + 2, lc + 2, rr - 2, cur_col - 2)
            split(cur_row + 2, cur_col + 2, rr - 2, rc - 2) 

        split(0, 0, self.height - 1, self.width - 1)
        # 在递归完成后设置起点和终点，确保它们与迷宫内部连通
        self.matrix[self.start[0], self.start[1]] = 0
        self.matrix[self.destination[0], self.destination[1]] = 0
        # 确保起点和终点周围有通路
        for d in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
            r, c = self.start[0] + d[0], self.start[1] + d[1]
            if 0 <= r < self.height and 0 <= c < self.width and self.matrix[r][c] == -1:
                self.matrix[r][c] = 0
            r, c = self.destination[0] + d[0], self.destination[1] + d[1]
            if 0 <= r < self.height and 0 <= c < self.width and self.matrix[r][c] == -1:
                self.matrix[r][c] = 0

    def generate_matrix_kruskal(self):
        """使用Kruskal算法生成迷宫地图"""
        self.matrix = -np.ones((self.height, self.width))

        def check(row, col):
            ans, counter = [], 0
            for d in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                row_, col_ = row + d[0], col + d[1]
                if row_ > 0 and row_ < self.height - 1 and col_ > 0 and col_ < self.width - 1 and self.matrix[row_, col_] == -1:
                    ans.append([d[0] * 2, d[1] * 2])
                    counter += 1
            if counter <= 1:
                return []
            return ans

        nodes = set()
        row = 1
        while row < self.height:
            col = 1
            while col < self.width:
                self.matrix[row, col] = 0
                nodes.add((row, col))
                col += 2
            row += 2

        unionset = UnionSet(nodes)
        # 如果nodes为空或只有一个节点，直接返回
        if len(nodes) == 0:
            # 对于太小的地图，使用简单连通方式
            self.matrix[self.start[0], self.start[1]] = 0
            self.matrix[self.destination[0], self.destination[1]] = 0
            # 创建简单路径
            for r in range(min(self.start[0], self.destination[0]), max(self.start[0], self.destination[0]) + 1):
                self.matrix[r, self.start[1]] = 0
            for c in range(min(self.start[1], self.destination[1]), max(self.start[1], self.destination[1]) + 1):
                self.matrix[self.destination[0], c] = 0
            return
            
        nodes_list = list(nodes)  # 转换为列表以便随机选择
        while unionset.count > 1 and len(nodes_list) > 0:
            idx = random.randint(0, len(nodes_list) - 1)
            row, col = nodes_list.pop(idx)
            directions = check(row, col)
            if len(directions):
                random.shuffle(directions)
                for d in directions:
                    row_, col_ = row + d[0], col + d[1]
                    if unionset.find((row, col)) == unionset.find((row_, col_)):
                        continue
                    nodes_list.append((row, col))  # 重新加入列表
                    unionset.count -= 1
                    unionset.union((row, col), (row_, col_))

                    if row == row_:
                        self.matrix[row][min(col, col_) + 1] = 0
                    else:
                        self.matrix[min(row, row_) + 1][col] = 0
                    break

        # 确保起点和终点与迷宫内部连通
        self.matrix[self.start[0], self.start[1]] = 0
        self.matrix[self.destination[0], self.destination[1]] = 0
        # 确保起点和终点周围有通路
        for d in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
            r, c = self.start[0] + d[0], self.start[1] + d[1]
            if 0 <= r < self.height and 0 <= c < self.width and self.matrix[r][c] == -1:
                self.matrix[r][c] = 0
            r, c = self.destination[0] + d[0], self.destination[1] + d[1]
            if 0 <= r < self.height and 0 <= c < self.width and self.matrix[r][c] == -1:
                self.matrix[r][c] = 0

    def find_path_dfs(self, destination):
        """使用DFS算法寻路"""
        visited = [[0 for i in range(self.width)] for j in range(self.height)]

        def dfs(path):
            visited[path[-1][0]][path[-1][1]] = 1
            if path[-1][0] == destination[0] and path[-1][1] == destination[1]:
                self.path = path[:]
                return
            for d in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                row_, col_ = path[-1][0] + d[0], path[-1][1] + d[1]
                if row_ > 0 and row_ < self.height - 1 and col_ > 0 and col_ < self.width and visited[row_][col_] == 0 and self.matrix[row_][col_] == 0:
                    dfs(path + [[row_, col_]])

        dfs([[self.start[0], self.start[1]]])

