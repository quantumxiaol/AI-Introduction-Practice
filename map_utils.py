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

    def generate_matrix(self, new_matrix=None):
        """生成地图矩阵"""
        self.generate_matrix_brick()

    def generate_matrix_brick(self):
        """使用预设砖块地图生成"""
        if self.height == 5:
            self.matrix = copy.copy(BRICK_MATRIX_5)
        else:
            self.matrix = copy.copy(BRICK_MATRIX_31)
        self.matrix[self.start[0]][self.start[1]] = 1
        self.matrix[self.destination[0]][self.destination[1]] = 2

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
            if row == self.start[0] and col == self.start[1] + 1:
                return

            directions = [[0, 2], [0, -2], [2, 0], [-2, 0]]
            random.shuffle(directions)
            for d in directions:
                row_, col_ = row + d[0], col + d[1]
                if row_ > 0 and row_ < self.height - 1 and col_ > 0 and col_ < self.width - 1 and visit_flag[row_][col_] == 0 and check(row, col, row_, col_):
                    if row == row_:
                        visit_flag[row][min(col, col_) + 1] = 1
                        self.matrix[row][min(col, col_) + 1] = 0
                    else:
                        visit_flag[min(row, row_) + 1][col] = 1
                        self.matrix[min(row, row_) + 1][col] = 0
                    dfs(row_, col_)

        dfs(self.destination[0], self.destination[1] - 1)
        self.matrix[self.start[0], self.start[1] + 1] = 0

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

