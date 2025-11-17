from PyQt6.QtWidgets import (QApplication, QMainWindow, QGraphicsView, 
                              QGraphicsScene, QGraphicsRectItem, QGraphicsEllipseItem,
                              QGraphicsTextItem, QLabel, QMenuBar, QMenu, QMessageBox)
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QColor, QPen, QBrush, QKeyEvent
import pandas as pd
import numpy as np
from PIL import Image
import time
import copy
import math
import os
import seaborn as sns
import random
from collections import deque
import queue
from queue import PriorityQueue

#                0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
brick_matrix=   [  
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
# brick_matrix = brick_matrix.reshape(31,31)

class UnionSet(object):
	"""
	并查集实现,构造函数中的matrix是一个numpy类型
	"""
	def __init__(self, arr):
		self.parent = {pos: pos for pos in arr}
		self.count = len(arr)

	def find(self, root):
		if root == self.parent[root]:
			return root
		return self.find(self.parent[root])

	def union(self, root1, root2):
		self.parent[self.find(root1)] = self.find(root2)

class Map:
    def __init__(self, width = 31, height = 31):

        assert width >= 5 and height >= 5, "Length of width or height must be larger than 5."
        self.width = (width // 2) * 2 + 1
        self.height = (height // 2) * 2 + 1
        self.start = [14, 15]
        self.destination = [self.height - 2, self.width - 2]
        self.matrix = None
        self.path = []

    def print_matrix(self):
        matrix = copy.deepcopy(self.matrix)
        for p in self.path:
            matrix[p[0]][p[1]] = 1
        for i in range(self.height):
            for j in range(self.width):
                if matrix[i][j] == -1:
                    print('B', end = '')
                elif matrix[i][j] == 0:
                    print('  ', end = '')
                elif matrix[i][j] == 1:
                    print('M', end = '')
                elif matrix[i][j] == 2:
                    print('F', end = '')
                elif matrix[i][j] == 3:
                    print('G', end = '')                    
            print('')

    def generate_matrix(self, new_matrix):
        # self.matrix = new_matrix
        # self.generate_matrix_dfs()
        # self.generate_matrix_prim()
        self.generate_matrix_brick()

    def resize_matrix(self, width, height, mode, new_matrix):
        self.path = []
        self.width = (width // 2) * 2 + 1
        self.height = (height // 2) * 2 + 1
        self.start = [14, 15]

        self.destination = [self.height - 2, self.width - 2]
        self.generate_matrix(mode, new_matrix)


    def generate_matrix_dfs(self):
        # 地图初始化
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

    def generate_matrix_brick(self):
        self.matrix = copy.copy(brick_matrix)
        self.matrix[self.start[0]][self.start[1]] = 1
        self.matrix[self.destination[0]][self.destination[1]] = 2
        for i in range(rows):
            print(self.matrix[i])

	# 迷宫寻路算法dfs
    def find_path_dfs(self, destination):
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

# 保存类引用，避免被全局变量覆盖
_MapClass = Map

def draw_pacman(scene, row, col, color='#B0E0E6'):
    x0, y0 = col * cell_width, row * cell_width
    x1, y1 = x0 + cell_width, y0 + cell_width
    # PyQt6 使用 QGraphicsEllipseItem 绘制扇形（通过设置起始角度和跨度）
    ellipse = QGraphicsEllipseItem(x0, y0, cell_width, cell_width)
    ellipse.setStartAngle(30 * 16)  # PyQt6 使用 1/16 度为单位
    ellipse.setSpanAngle(300 * 16)
    ellipse.setBrush(QBrush(QColor(color)))
    ellipse.setPen(QPen(QColor('yellow'), 0))
    scene.addItem(ellipse)

def draw_cell(scene, row, col, color="#F2F2F2"):
    x0, y0 = col * cell_width, row * cell_width
    x1, y1 = x0 + cell_width, y0 + cell_width
    rect = QGraphicsRectItem(x0, y0, cell_width, cell_width)
    rect.setBrush(QBrush(QColor(color)))
    rect.setPen(QPen(QColor(color), 0))
    scene.addItem(rect)

def draw_food(scene, row, col, color="#EE3F4D"):
    x0, y0 = col * cell_width, row * cell_width
    x1, y1 = x0 + cell_width, y0 + cell_width
    rect = QGraphicsRectItem(x0, y0, cell_width, cell_width)
    rect.setBrush(QBrush(QColor(color)))
    rect.setPen(QPen(QColor(color), 0))
    scene.addItem(rect)

def draw_path(scene, matrix, row, col, color, line_color):
    # 列
    if row + 1 < rows and matrix[row - 1][col] >= 1 and matrix[row + 1][col] >= 1:
        x0, y0 = col * cell_width + 2 * cell_width / 5, row * cell_width
        x1, y1 = x0 + cell_width / 5, y0 + cell_width
    # 行
    elif col + 1 < cols and matrix[row][col - 1] >= 1 and matrix[row][col + 1] >= 1:
        x0, y0 = col * cell_width, row * cell_width + 2 * cell_width / 5
        x1, y1 = x0 + cell_width, y0 + cell_width / 5
    # 左上角
    elif col + 1 < cols and row + 1 < rows and matrix[row][col + 1] >= 1 and matrix[row + 1][col] >= 1:
        x0, y0 = col * cell_width + 2 * cell_width / 5, row * cell_width + 2 * cell_width / 5
        x1, y1 = x0 + 3 * cell_width / 5, y0 + cell_width / 5
        rect = QGraphicsRectItem(x0, y0, x1 - x0, y1 - y0)
        rect.setBrush(QBrush(QColor(color)))
        rect.setPen(QPen(QColor(line_color), 0))
        scene.addItem(rect)
        x0, y0 = col * cell_width + 2 * cell_width / 5, row * cell_width + 2 * cell_width / 5
        x1, y1 = x0 + cell_width / 5, y0 + 3 * cell_width / 5
    # 右上角
    elif row + 1 < rows and matrix[row][col - 1] >= 1 and matrix[row + 1][col] >= 1:
        x0, y0 = col * cell_width, row * cell_width + 2 * cell_width / 5
        x1, y1 = x0 + 3 * cell_width / 5, y0 + cell_width / 5
        rect = QGraphicsRectItem(x0, y0, x1 - x0, y1 - y0)
        rect.setBrush(QBrush(QColor(color)))
        rect.setPen(QPen(QColor(line_color), 0))
        scene.addItem(rect)
        x0, y0 = col * cell_width + 2 * cell_width / 5, row * cell_width + 2 * cell_width / 5
        x1, y1 = x0 + cell_width / 5, y0 + 3 * cell_width / 5
    # 左下角
    elif col + 1 < cols and matrix[row - 1][col] >= 1 and matrix[row][col + 1] >= 1:
        x0, y0 = col * cell_width + 2 * cell_width / 5, row * cell_width
        x1, y1 = x0 + cell_width / 5, y0 + 3 * cell_width / 5
        rect = QGraphicsRectItem(x0, y0, x1 - x0, y1 - y0)
        rect.setBrush(QBrush(QColor(color)))
        rect.setPen(QPen(QColor(line_color), 0))
        scene.addItem(rect)
        x0, y0 = col * cell_width + 2 * cell_width / 5, row * cell_width + 2 * cell_width / 5
        x1, y1 = x0 + 3 * cell_width / 5, y0 + cell_width / 5
    # 右下角
    elif matrix[row - 1][col] >= 1 and matrix[row][col - 1] >= 1:
        x0, y0 = col * cell_width, row * cell_width + 2 * cell_width / 5
        x1, y1 = x0 + 3 * cell_width / 5, y0 + cell_width / 5
        rect = QGraphicsRectItem(x0, y0, x1 - x0, y1 - y0)
        rect.setBrush(QBrush(QColor(color)))
        rect.setPen(QPen(QColor(line_color), 0))
        scene.addItem(rect)
        x0, y0 = col * cell_width + 2 * cell_width / 5, row * cell_width
        x1, y1 = x0 + cell_width / 5, y0 + 3 * cell_width / 5
    else:
        x0, y0 = col * cell_width + 2 * cell_width / 5, row * cell_width + 2 * cell_width / 5
        x1, y1 = x0 + cell_width / 5, y0 + cell_width / 5
    rect = QGraphicsRectItem(x0, y0, x1 - x0, y1 - y0)
    rect.setBrush(QBrush(QColor(color)))
    rect.setPen(QPen(QColor(line_color), 0))
    scene.addItem(rect)

def draw_map(scene, matrix, path, moves):
    """
    根据matrix中每个位置的值绘图
    -1: 墙壁
    0 : 空白
    1 : pacman
    2 : food
    3 : ghost
    """

    scene.clear()
    matrix = copy.copy(matrix)
    
    # for p in path:
    #     matrix[p[0]][p[1]] = 1
    # for move in moves:
    #     matrix[move[0]][move[1]] = 2
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 0:
                draw_cell(scene, r, c)
            elif matrix[r][c] == -1:
                draw_cell(scene, r, c, '#525288')
            elif matrix[r][c] == 1:

                # draw_cell(scene, r, c)
                draw_pacman(scene,r,c)
                # draw_path(scene, matrix, r, c, '#bc84a8', '#bc84a8')
            elif matrix[r][c] == 2:
                draw_food(scene, r, c)
                # draw_path(canvas, matrix, r, c, '#ee3f4d', '#ee3f4d')

def update_map(scene, matrix, path, moves):
    
    scene.clear()
    matrix = copy.copy(matrix)
    # for p in path:
    #     matrix[p[0]][p[1]] = 1
    # for move in moves:
    #     matrix[move[0]][move[1]] = 2

    row, col = movement_list[-1]
    colors = ['#525288', '#F2F2F2', '#525288', '#F2F2F2', '#525288', '#F2F2F2', '#525288', '#F2F2F2']
    
    for r in range(rows):
        for c in range(cols):
            # distance = (row - r) * (row - r) + (col - c) * (col - c)
            # if distance >= 100:
            #     color = colors[0:2]
            # elif distance >= 60:
            #     color = colors[2:4]
            # elif distance >= 30:
            #     color = colors[4:6]
            # else:
            #     color = colors[6:8]

            if matrix[r][c] == 0:
                draw_cell(scene, r, c, colors[1])
            elif matrix[r][c] == -1:
                draw_cell(scene, r, c, colors[0])
            elif matrix[r][c] == 1:
                draw_pacman(scene,r,c)
                # draw_cell(scene, r, c, colors[1])
                # draw_path(scene, matrix, r, c, '#bc84a8', '#bc84a8')
            elif matrix[r][c] == 2:
                draw_food(scene, r, c)
                # draw_cell(canvas, r, c, colors[1])
                # draw_path(canvas, matrix, r, c, '#ee3f4d', '#ee3f4d')
    

def generate_matrix():
    global movement_list
    global click_counter, back_counter
    global visited_paths, final_path

    click_counter, back_counter = 0, 0
    movement_list = [Map.start]  # 重置移动列表
    Map.path = []  # 清空路径
    visited_paths = set()  # 清空已访问路径
    final_path = []  # 清空最终路径
    # 重新生成地图
    Map.generate_matrix(None)
    # 确保起点和终点正确设置（覆盖任何之前的修改）
    Map.matrix[Map.start[0]][Map.start[1]] = 1
    Map.matrix[Map.destination[0]][Map.destination[1]] = 2
    # 确保地图中其他位置没有被标记为已访问
    for r in range(Map.height):
        for c in range(Map.width):
            if Map.matrix[r][c] == -2 or Map.matrix[r][c] == -3:
                # 恢复被标记为已访问或搜索中的位置
                if (r, c) != (Map.start[0], Map.start[1]) and (r, c) != (Map.destination[0], Map.destination[1]):
                    # 检查原始地图中这个位置应该是什么
                    if brick_matrix[r][c] == -1:
                        Map.matrix[r][c] = -1
                    else:
                        Map.matrix[r][c] = 0
    # 再次确保起点和终点正确
    Map.matrix[Map.start[0]][Map.start[1]] = 1
    Map.matrix[Map.destination[0]][Map.destination[1]] = 2
    draw_map(scene, Map.matrix, Map.path, movement_list)

# 全局变量存储已访问的路径和最终路径
visited_paths = set()
final_path = []

def update_map_search(scene, matrix, path, moves, visited=None, final_path_points=None):
    """更新地图显示，高亮当前搜索位置和路径"""
    global visited_paths, final_path
    
    scene.clear()
    # 使用原始地图的深拷贝，避免修改原始地图
    display_matrix = copy.deepcopy(Map.matrix)
    
    # 更新已访问路径
    if visited:
        visited_paths.update(visited)
    if final_path_points:
        final_path = final_path_points.copy()
    
    # 标记已访问的路径（用浅蓝色）
    for pos in visited_paths:
        if isinstance(pos, tuple) and len(pos) == 2:
            r, c = pos
            if (r, c) != (Map.start[0], Map.start[1]) and (r, c) != (Map.destination[0], Map.destination[1]):
                display_matrix[r][c] = -4  # 已访问标记
    
    # 标记最终路径（用绿色）
    for pos in final_path:
        if isinstance(pos, tuple) and len(pos) == 2:
            r, c = pos
            if (r, c) != (Map.start[0], Map.start[1]) and (r, c) != (Map.destination[0], Map.destination[1]):
                display_matrix[r][c] = -5  # 最终路径标记
    
    # 标记当前搜索位置
    if isinstance(moves, tuple) and len(moves) == 2:
        display_matrix[moves[0]][moves[1]] = -3
    
    # 显示起点和终点（确保它们始终显示）
    display_matrix[Map.start[0]][Map.start[1]] = 1
    display_matrix[Map.destination[0]][Map.destination[1]] = 2

    # 使用起点作为默认位置，不依赖 movement_list
    row, col = Map.start
    colors = ['#525288', '#F2F2F2', '#525288', '#F2F2F2', '#525288', '#F2F2F2', '#525288', '#F2F2F2']
    
    for r in range(rows):
        for c in range(cols):
            if display_matrix[r][c] == 0:
                draw_cell(scene, r, c, colors[1])
            elif display_matrix[r][c] == -1:
                draw_cell(scene, r, c, colors[0])
            elif display_matrix[r][c] == -2:
                draw_cell(scene, r, c, "#CCCCCC")  # 已访问过的路径用浅灰色
            elif display_matrix[r][c] == -4:
                draw_cell(scene, r, c, "#ADD8E6")  # 已访问路径用浅蓝色
            elif display_matrix[r][c] == -5:
                draw_cell(scene, r, c, "#90EE90")  # 最终路径用浅绿色
            elif display_matrix[r][c] == 1:
                draw_pacman(scene, r, c)
            elif display_matrix[r][c] == 2:
                draw_food(scene, r, c)
            elif display_matrix[r][c] == -3:
                draw_cell(scene, r, c, "#FFD700")  # 当前搜索位置用金色高亮
    # windows.after(500,update_map_search(canvas, matrix, path, moves))
    # time.sleep(0.1)
    # print("called\n")
    # matrix[moves[0]][moves[1]] = 0

def movement_update_handler(event):
    global movement_list
    global click_counter, back_counter

    cur_pos = movement_list[-1]

    ops = {'Left': [0, -1], 'Right': [0, 1], 'Up': [-1, 0], 'Down': [1, 0], 'a': [0, -1], 'd': [0, 1], 'w': [-1, 0], 's': [1, 0]}
    r_, c_ = cur_pos[0] + ops[event.keysym][0], cur_pos[1] + ops[event.keysym][1]
    if len(movement_list) > 1 and [r_, c_] == movement_list[-2]:
        click_counter += 1
        back_counter += 1
        movement_list.pop()
        if  r_ < Map.height and c_ < Map.width and Map.matrix[r_][c_] == 0:
            click_counter += 1
        movement_list.append([r_, c_])
    Map.path = []

    # 可以移动
    if Map.matrix[cur_pos[0]+ops[event.keysym][0]][cur_pos[1]+ops[event.keysym][1]] == 0:
        Map.matrix[cur_pos[0]][cur_pos[1]] = 0
        Map.matrix[cur_pos[0]+ops[event.keysym][0]][cur_pos[1]+ops[event.keysym][1]] = 1
        movement_list.append([r_, c_])
    elif Map.matrix[cur_pos[0]+ops[event.keysym][0]][cur_pos[1]+ops[event.keysym][1]] == 2:
        Map.matrix[cur_pos[0]][cur_pos[1]] = 0
        Map.matrix[cur_pos[0]+ops[event.keysym][0]][cur_pos[1]+ops[event.keysym][1]] = 1
        movement_list.append([r_, c_])
    # 不可以移动    
    elif Map.matrix[cur_pos[0]+ops[event.keysym][0]][cur_pos[1]+ops[event.keysym][1]] ==-1:
        Map.matrix[cur_pos[0]][cur_pos[1]] = 1

    # Map.matrix[cur_pos[0]][cur_pos[1]]=0
    # Map.matrix[cur_pos[0]+ops[event.keysym][0]][cur_pos[1]+ops[event.keysym][1]]=1

    update_map(scene, Map.matrix, Map.path, movement_list)
    check_reach()

dirs = [
    lambda x,y :(x+1,y),
    lambda x,y :(x-1,y),
    lambda x,y :(x,y-1),
    lambda x,y :(x,y+1)
]

def getSuccessors(cur_pos):
    able=[]
    row, col = cur_pos[0], cur_pos[1]
    # 上方 - 检查边界
    if row > 0 and (Map.matrix[row-1][col] == 0 or Map.matrix[row-1][col] == 2):
        able.append(((row-1, col),[-1,0],1))
    # 下方 - 检查边界
    if row < Map.height - 1 and (Map.matrix[row+1][col] == 0 or Map.matrix[row+1][col] == 2):
        able.append(((row+1, col),[1,0],1))       
    # 左方 - 检查边界
    if col > 0 and (Map.matrix[row][col-1] == 0 or Map.matrix[row][col-1] == 2):
        able.append(((row, col-1),[0,-1],1))
    # 右方 - 检查边界
    if col < Map.width - 1 and (Map.matrix[row][col+1] == 0 or Map.matrix[row][col+1] == 2):
        able.append(((row, col+1),[0,1],1))
    return able


def movement_dfs():
    global movement_list
    global click_counter, back_counter
    global visited_paths, final_path

    # 清空之前的路径记录
    visited_paths = set()
    final_path = []
    
    # 使用地图副本进行搜索，避免修改原始地图
    search_matrix = copy.deepcopy(Map.matrix)
    end = (Map.destination[0],Map.destination[1])
    start = (Map.start[0],Map.start[1])
    path_list = [start]
    visited_set = set()  # 记录已访问的节点

    # path_list=[cur_pos]
    while path_list:
        cur_pos = path_list[-1]
        if cur_pos == end:
            print(path_list)
            print("RunOutSuccessfly\n")
            final_path = path_list.copy()  # 保存最终路径
            break
        row , col = cur_pos
        # 已经走过
        search_matrix[row][col] = -2
        visited_set.add(cur_pos)  # 记录已访问
        # 上方 - 检查边界
        if row > 0 and (search_matrix[row-1][col] == 0 or search_matrix[row-1][col] == 2):
            path_list.append((row-1, col))
            search_matrix[row-1][col] = 1
            continue
        # 下方 - 检查边界
        elif row < Map.height - 1 and (search_matrix[row+1][col] == 0 or search_matrix[row+1][col] == 2):
            path_list.append((row+1, col))
            search_matrix[row+1][col] = 1
            continue        
        # 左方 - 检查边界
        elif col > 0 and (search_matrix[row][col-1] == 0 or search_matrix[row][col-1] == 2):
            path_list.append((row, col-1))
            search_matrix[row][col-1] = 1
            continue        
        # 右方 - 检查边界
        elif col < Map.width - 1 and (search_matrix[row][col+1] == 0 or search_matrix[row][col+1] == 2):
            path_list.append((row, col+1))
            search_matrix[row][col+1] = 1
            continue        
        else: 
            path_list.pop()
    else:
        print("Error\n")
        
    # 显示完整路径
    for p in path_list:
        update_map_search(scene, Map.matrix, Map.path, p, visited=visited_set, final_path_points=final_path)
        time.sleep(0.05)  # 添加延迟以便看到动画
        QApplication.processEvents()  # 处理事件以更新界面
        
    # check_reach()

def movement_bfs():
    global movement_list
    global click_counter, back_counter
    global visited_paths, final_path

    # 清空之前的路径记录
    visited_paths = set()
    final_path = []
    
    # 使用地图副本进行搜索，避免修改原始地图
    search_matrix = copy.deepcopy(Map.matrix)
    end = (Map.destination[0],Map.destination[1])
    start = (Map.start[0],Map.start[1])
    cur_pos = movement_list[-1]
    path_list = []
    visited_set = set()  # 记录已访问的节点
    #创建队列 起点入队,起点没有上一节点所里这里的联系用-1表示
    queue = deque()
    queue.append((start[0],start[1],-1))
    while len(queue)>0:
        curnode = queue.popleft()
        path_list.append(curnode)
        visited_set.add((curnode[0], curnode[1]))  # 记录已访问
        #找到迷宫终点跳出循环
        if curnode[0] == end[0] and curnode[1] == end[1]:
            cur = path_list[-1]
            #存放最终路径结果
            path_result = []
            while cur[2] != -1:#只有起点的第三个元素才是-1
                path_result.append((cur[0],cur[1]))#路径不用储存节点之间的联系
                cur = path_list[cur[2]]#找到上一节点
            path_result.reverse()
            print(path_result)
            # for path in path_result:
                # print(path)
                # print(path_list)
            print("RunOutSuccessfly\n")
            final_path = path_result.copy()  # 保存最终路径
            # 显示完整路径
            for p in path_result:
                update_map_search(scene, Map.matrix, Map.path, p, visited=visited_set, final_path_points=final_path)
                time.sleep(0.05)  # 添加延迟以便看到动画
                QApplication.processEvents()  # 处理事件以更新界面
            return True    
        #未找到终点执行循环
        for dir in dirs:
            nextnode = dir(curnode[0],curnode[1])
            #判断下一节点是否可通过
            if search_matrix[nextnode[0]][nextnode[1]] == 0 or search_matrix[nextnode[0]][nextnode[1]] == 2:
                #队列元素与nextnode形式不同，队列中要加入节点间的联系，上面知道联系储存在path_list中
                queue.append((nextnode[0],nextnode[1],path_list.index(curnode)))
                #将循环过的节点标记为走过
                search_matrix[nextnode[0]][nextnode[1]] = -2
                #这里不能break 因为最广是探索所有的路径 所以要找到所有可通过的 最深的只要找到一个就可以了 所以栈那里需要break
    else:
        print("Error")

def movement_ucs():
    global movement_list
    global click_counter, back_counter
    global visited_paths, final_path

    # 清空之前的路径记录
    visited_paths = set()
    final_path = []
    
    end = (Map.destination[0],Map.destination[1])
    start = (Map.start[0],Map.start[1])
    cur_pos = start
    path = [(start)]
    # 初始化相关参数
    result = []
    explored = set()
    visited_set = set()  # 记录已访问的节点
    frontier = queue.PriorityQueue()
    # 定义起始状态，其中包括开始的位置，对应的行动方案和行动代价
    start = ((Map.start[0],Map.start[1]), [], 0)
    # print (start)
    # 把起始状态放进frontier队列中，update方法会自动对其中的状态按照其行动代价进行排序
    frontier.put(start,0)
    # 构造循环，循环读取frontier中的状态，进行判定
    while not frontier.empty():
        # 获取当前节点的各项信息
        (node, move, cost) = frontier.get()
        visited_set.add(node)  # 记录已访问
        # 如果弹出的节点状态满足目标要求，停止循环
        if node == (end[0],end[1]):
            result = move
            
            break
        # 如果该节点该节点不满足目标要求，判定其是否访问过
        if node not in explored:
            explored.add(node)
            # 遍历这个节点的子节点，更新frontier队列
            for child,direction,step in getSuccessors(node):
                newMove = move + [direction]
                newCost = cost + step
                newNode = (child, newMove, newCost)
                frontier.put(newNode, newCost)
    # 返回计算结果，即一个行动方案
    if not result:
        print("Error: No path found\n")
        return result
    
    # 构建最终路径
    final_path_list = [(Map.start[0], Map.start[1])]
    temp_pos = (Map.start[0], Map.start[1])
    for p in result:
        temp_pos = (temp_pos[0]+p[0], temp_pos[1]+p[1])
        final_path_list.append(temp_pos)
    final_path = final_path_list.copy()
    
    # 显示完整路径
    for p in result:
        cur_pos = (cur_pos[0]+p[0],cur_pos[1]+p[1])
        path.append(cur_pos)
        update_map_search(scene, Map.matrix, Map.path, cur_pos, visited=visited_set, final_path_points=final_path)
        time.sleep(0.05)  # 添加延迟以便看到动画
        QApplication.processEvents()  # 处理事件以更新界面
    print("RunOutSuccessfly\n")
    print(path)
    return result

def movement_astar():
    global visited_paths, final_path

    # 清空之前的路径记录
    visited_paths = set()
    final_path = []
    
    # 使用地图副本进行搜索，避免修改原始地图
    lab = copy.deepcopy(Map.matrix)
    end = (Map.destination[0],Map.destination[1])
    start = (Map.start[0],Map.start[1])
    (i_s, j_s) = start
    # and take the goal position (used in the heuristic)
    (i_e, j_e) = end

    width = len(lab[0])
    height = len(lab)

    heuristic = lambda i, j: abs(i_e - i) + abs(j_e - j)
    comp = lambda state: state[2] + state[3] # get the total cost

    # small variation for easier code, state is (coord_tuple, previous, path_cost, heuristic_cost)
    fringe = [((i_s, j_s), list(), 0, heuristic(i_s, j_s))]
    visited = {} # empty set
    visited_set = set()  # 记录已访问的节点

    while fringe:
        # get first state (least cost)
        state = fringe.pop(0)
        # print(state)
        # goal check
        (i, j) = state[0]
        visited_set.add((i, j))  # 记录已访问
        # print(state[0])
        
        if (i, j) == end:
            path = [state[0]] + state[1]
            path.reverse()
            print(path)
            print("RunOutSuccessfly\n")
            final_path = path.copy()  # 保存最终路径
            # 显示完整路径
            for p in path:
                update_map_search(scene, Map.matrix, Map.path, (p[0],p[1]), visited=visited_set, final_path_points=final_path)
                time.sleep(0.05)  # 添加延迟以便看到动画
                QApplication.processEvents()  # 处理事件以更新界面
            return path

        # set the cost (path is enough since the heuristic won't change)
        visited[(i, j)] = state[2] 

        # explore neighbor
        neighbor = list()
        if i > 0 and lab[i-1][j] >= 0 : 
            neighbor.append((i-1, j))
        if i < height - 1 and lab[i+1][j] >= 0 :
            neighbor.append((i+1, j))
        if j > 0 and lab[i][j-1] >= 0 :
            neighbor.append((i, j-1))
        if j < width - 1 and lab[i][j+1] >= 0 :
            neighbor.append((i, j+1))

        for n in neighbor:
            next_cost = state[2] + 1
            if n in visited and visited[n] >= next_cost:
                continue
            fringe.append((n, [state[0]] + state[1], next_cost, heuristic(n[0], n[1])))

        # resort the list (SHOULD use a priority queue here to avoid re-sorting all the time)
        fringe.sort(key=comp)
    
    # 如果没有找到路径
    print("Error: No path found\n")
    return []

def movement_a():
    global movement_list
    global click_counter, back_counter

    cur_pos = movement_list[-1]

    update_map(scene, Map.matrix, Map.path, movement_list)
    check_reach()

def check_reach():
    global next_Map_flag
    if movement_list[-1] == Map.destination:
        print("Congratulations! You Eat all the food! {}".format(back_counter))
        x0, y0 = cols * cell_width / 2 - 200, 30
        x1, y1 = x0 + 400, y0 + 40
        rect = QGraphicsRectItem(x0, y0, x1 - x0, y1 - y0)
        rect.setBrush(QBrush(QColor('#F2F2F2')))
        rect.setPen(QPen(QColor('#525288'), 3))
        scene.addItem(rect)
        text = QGraphicsTextItem("Congratulations! You Eat all the food! ")
        text.setPos(cols * cell_width / 2 - 150, y0 + 20)
        text.setDefaultTextColor(QColor("#525288"))
        scene.addItem(text)
        next_Map_flag = True
        return True

class PacmanWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        global scene, view, status_label
        
        self.setWindowTitle("PACMAN")
        # 为菜单栏和状态栏预留空间
        menu_height = self.menuBar().height() if self.menuBar() else 25
        status_height = 30
        self.setFixedSize(width, height + menu_height + status_height)
        
        # 创建菜单栏（必须在设置窗口大小之前）
        menubar = self.menuBar()
        # 在 macOS 上强制显示菜单栏在窗口内（而不是系统菜单栏）
        try:
            menubar.setNativeMenuBar(False)
        except:
            pass  # 如果方法不存在则忽略
        filemenu = menubar.addMenu('设置')
        dfs_action = filemenu.addAction('深度优先', movement_dfs)
        dfs_action.setShortcut('F1')
        bfs_action = filemenu.addAction('广度优先', movement_bfs)
        bfs_action.setShortcut('F2')
        ucs_action = filemenu.addAction('一致代价', movement_ucs)
        ucs_action.setShortcut('F3')
        astar_action = filemenu.addAction('A*', movement_astar)
        astar_action.setShortcut('F4')
        filemenu.addSeparator()
        exit_action = filemenu.addAction('退出', self.close)
        exit_action.setShortcut('F5')
        restart_action = filemenu.addAction('重新开始', generate_matrix)
        restart_action.setShortcut('F9')
        
        # 创建场景和视图
        scene = QGraphicsScene()
        scene.setSceneRect(0, 0, width, height)
        scene.setBackgroundBrush(QBrush(QColor("#F2F2F2")))
        
        view = QGraphicsView(scene, self)
        view.setFixedSize(width, height)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setCentralWidget(view)
        
        # 创建状态栏
        status_label = QLabel("PAC Man")
        self.statusBar().addWidget(status_label)
        
        # 初始化游戏
        global Map, movement_list
        # 使用保存的类引用来创建实例
        Map = _MapClass(cols, rows)
        movement_list = [Map.start]
        generate_matrix()
        
    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Left or key == Qt.Key.Key_A:
            movement_update_handler(type('obj', (object,), {'keysym': 'Left'})())
        elif key == Qt.Key.Key_Right or key == Qt.Key.Key_D:
            movement_update_handler(type('obj', (object,), {'keysym': 'Right'})())
        elif key == Qt.Key.Key_Up or key == Qt.Key.Key_W:
            movement_update_handler(type('obj', (object,), {'keysym': 'Up'})())
        elif key == Qt.Key.Key_Down or key == Qt.Key.Key_S:
            movement_update_handler(type('obj', (object,), {'keysym': 'Down'})())
        elif key == Qt.Key.Key_F1:
            movement_dfs()
        elif key == Qt.Key.Key_F2:
            movement_bfs()
        elif key == Qt.Key.Key_F3:
            movement_ucs()
        elif key == Qt.Key.Key_F4:
            movement_astar()
        elif key == Qt.Key.Key_F5:
            self.close()
        elif key == Qt.Key.Key_F9:
            generate_matrix()

if __name__ == '__main__':
    # 基础参数
    cell_width = 20
    rows = 31
    cols = 31
    height = cell_width * rows
    width = cell_width * cols

    click_counter, total_counter, back_counter = 0, 0, 0
    scene = None
    view = None
    status_label = None
    Map = None
    movement_list = None
    next_Map_flag = False
    
    t0 = int(time.time())
    t1 = t0

    app = QApplication([])
    windows = PacmanWindow()
    windows.show()
    app.exec()