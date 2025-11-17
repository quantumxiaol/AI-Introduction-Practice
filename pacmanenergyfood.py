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
import numpy
import math

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

brick_matrix=   [  
                [-1,-1,-1,-1,-1],
                [-1, 0, 0, 0,-1],
                [-1, 0, 0, 0,-1],
                [-1, 0, 0, 0,-1],
                [-1,-1,-1,-1,-1]]

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
    def __init__(self, width = 5, height = 5):
        assert width >= 5 and height >= 5, "Length of width or height must be larger than 5."
        self.width = (width // 2) * 2 + 1
        self.height = (height // 2) * 2 + 1
        self.start = [1, 1]
        self.destination = [self.height - 2, self.width - 3]
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
        self.matrix = brick_matrix
        self.matrix[self.start[0]][self.start[1]] = 1
        self.matrix[self.destination[0]][self.destination[1]] = 2

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

# 保存 Map 类的引用，避免全局变量冲突
_MapClass = Map

def draw_pacman(scene, row, col, color='#B0E0E6'):
    x0, y0 = col * cell_width, row * cell_width
    x1, y1 = x0 + cell_width, y0 + cell_width
    ellipse = QGraphicsEllipseItem(x0, y0, cell_width, cell_width)
    ellipse.setStartAngle(30 * 16)
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
        rect = QGraphicsRectItem(x0, y0, x1 - x0, y1 - y0)
        rect.setBrush(QBrush(QColor(color)))
        rect.setPen(QPen(QColor(line_color), 0))
        scene.addItem(rect)
    # 行
    elif col + 1 < cols and matrix[row][col - 1] >= 1 and matrix[row][col + 1] >= 1:
        x0, y0 = col * cell_width, row * cell_width + 2 * cell_width / 5
        x1, y1 = x0 + cell_width, y0 + cell_width / 5
        rect = QGraphicsRectItem(x0, y0, x1 - x0, y1 - y0)
        rect.setBrush(QBrush(QColor(color)))
        rect.setPen(QPen(QColor(line_color), 0))
        scene.addItem(rect)
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
        rect = QGraphicsRectItem(x0, y0, x1 - x0, y1 - y0)
        rect.setBrush(QBrush(QColor(color)))
        rect.setPen(QPen(QColor(line_color), 0))
        scene.addItem(rect)
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
        rect = QGraphicsRectItem(x0, y0, x1 - x0, y1 - y0)
        rect.setBrush(QBrush(QColor(color)))
        rect.setPen(QPen(QColor(line_color), 0))
        scene.addItem(rect)
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
        rect = QGraphicsRectItem(x0, y0, x1 - x0, y1 - y0)
        rect.setBrush(QBrush(QColor(color)))
        rect.setPen(QPen(QColor(line_color), 0))
        scene.addItem(rect)
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
        rect = QGraphicsRectItem(x0, y0, x1 - x0, y1 - y0)
        rect.setBrush(QBrush(QColor(color)))
        rect.setPen(QPen(QColor(line_color), 0))
        scene.addItem(rect)
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
                draw_pacman(scene, r, c)
                # draw_path(scene, matrix, r, c, '#bc84a8', '#bc84a8')
            elif matrix[r][c] == 2:
                draw_food(scene, r, c)
                # draw_path(scene, matrix, r, c, '#ee3f4d', '#ee3f4d')

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
                draw_pacman(scene, r, c)
                # draw_cell(scene, r, c, colors[1])
                # draw_path(scene, matrix, r, c, '#bc84a8', '#bc84a8')
            elif matrix[r][c] == 2:
                draw_food(scene, r, c)
                # draw_cell(scene, r, c, colors[1])
                # draw_path(scene, matrix, r, c, '#ee3f4d', '#ee3f4d')
    

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
    draw_map(scene, Map.matrix, Map.path, movement_list)

def update_map_search(scene, matrix, path, moves, visited=None, final_path_points=None):
    global visited_paths, final_path
    
    scene.clear()
    # 使用原始地图的深拷贝，确保不修改原始地图
    display_matrix = copy.deepcopy(Map.matrix)
    
    # 使用传入的参数或全局变量
    if visited is None:
        visited = visited_paths if 'visited_paths' in globals() else set()
    if final_path_points is None:
        final_path_points = final_path if 'final_path' in globals() else []
    
    # 标记已访问的路径（用浅蓝色）
    for pos in visited:
        if isinstance(pos, tuple) and len(pos) == 2:
            r, c = pos
            if (r, c) != (Map.start[0], Map.start[1]) and (r, c) != (Map.destination[0], Map.destination[1]):
                display_matrix[r][c] = -4  # 已访问标记
    
    # 标记最终路径（用绿色）
    for pos in final_path_points:
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
    # windows.after(500,update_map_search(scene, matrix, path, moves))
    # time.sleep(0.1)
    # print("called\n")

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
    if Map.matrix[cur_pos[0]-1][cur_pos[1]] == 0 or Map.matrix[cur_pos[0]-1][cur_pos[1]] == 2:
        able.append(((cur_pos[0]-1,cur_pos[1]),[-1,0],1))

    # 下方
    if Map.matrix[cur_pos[0]+1][cur_pos[1]] == 0 or Map.matrix[cur_pos[0]+1][cur_pos[1]] == 2:
        able.append(((cur_pos[0]+1,cur_pos[1]),[1,0],1))       
    # 左方
    if Map.matrix[cur_pos[0]][cur_pos[1]-1] == 0 or Map.matrix[cur_pos[0]][cur_pos[1]-1] == 2:
        able.append(((cur_pos[0],cur_pos[1]-1),[0,-1],1))
        
    # 右方
    if Map.matrix[cur_pos[0]][cur_pos[1]+1] == 0 or Map.matrix[cur_pos[0]][cur_pos[1]+1] == 2:
        able.append(((cur_pos[0],cur_pos[1]+1),[0,1],1))
    return able


def movement_astar():
    global visited_paths, final_path

    # 清空之前的路径记录
    visited_paths = set()
    final_path = []
    
    lab=copy.deepcopy(Map.matrix)
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
            visited_paths = visited_set.copy()  # 保存已访问的节点
            # 显示完整路径
            for p in path:
                update_map_search(scene, Map.matrix, Map.path, (p[0],p[1]), visited=visited_set, final_path_points=final_path)
                time.sleep(0.05)
                QApplication.processEvents()
            return path

        # set the cost (path is enough since the heuristic won't change)
        visited[(i, j)] = state[2] 

        # explore neighbor
        neighbor = list()
        if i > 0 and lab[i-1][j] >= 0 : 
            neighbor.append((i-1, j))
        if i < height and lab[i+1][j] >= 0 :
            neighbor.append((i+1, j))
        if j > 0 and lab[i][j-1] >= 0 :
            neighbor.append((i, j-1))
        if j < width and lab[i][j+1] >= 0 :
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
    visited_paths = visited_set.copy()
    return None

def movement_a():
    global movement_list
    global click_counter, back_counter

    cur_pos = movement_list[-1]

    update_map(scene, Map.matrix, Map.path, movement_list)
    check_reach()

# 给定状态的Agent行为概率分布，RV
# （状态 上移概率、下移概率、左移概率、右移概率）
states = [   (0,0,0.5,0,0.5),(1,0,0.4,0.3,0.3),(2,0,0.5,0.5,0),
            (3,0.3,0.3,0,0.4),(4,0.25,0.25,0.25,0.25),(5,0.3,0.3,0.4,0),
            (6,0.5,0,0,0.5),(7,0.4,0,0.3,0.3),(8,0.5,0,0.5,0)]

# 参数
REWARD = -1 # constant reward for non-terminal states
DISCOUNT = 1
MAX_ERROR = 10**(-3)


NUM_ACTIONS = 4
ACTIONS = [(1, 0), (0, -1), (-1, 0), (0, 1)] # Down, Left, Up, Right
NUM_ROW = 3
NUM_COL = 3
U = [   [0, 0, 0],
        [0, 0, 0],
        [0, 1, 0]]

def printEP(arr, policy=False):
    res = ""
    for r in range(NUM_ROW):
        res += "|"
        for c in range(NUM_COL):
            # if r == c == 1:
            #     val = "WALL"

            if r == 2 and c == 1:
                val = "+1"

            else:
                if policy:
                    val = ["Down", "Left", "Up", "Right"][arr[r][c]]
                else:
                    val = str(arr[r][c])
            res += " " + val[:5].ljust(5) + " |" # format
        res += "\n"
    print(res)

# 获得通过从给定状态执行给定动作所达到的状态的效用
def getU(U, r, c, action):
    dr, dc = ACTIONS[action]
    newR, newC = r+dr, c+dc

    if newR < 0 or newC < 0 or newR >= NUM_ROW or newC >= NUM_COL : # boundary or the wallor (newR == newC == 1)
        return U[r][c]
    else:
        return U[newR][newC]

# 计算给定动作状态的效用
def calculateU(U, r, c, action):
    u = REWARD
    u += 0.1 * DISCOUNT * getU(U, r, c, (action-1)%4)
    u += 0.8 * DISCOUNT * getU(U, r, c, action)
    u += 0.1 * DISCOUNT * getU(U, r, c, (action+1)%4)
    return u

def valueIteration(U):
    print("值迭代:\n")
    while True:
        nextU =[[0, 0, 0],
                [0, 0, 0],
                [0, 1, 0]]
        error = 0
        for r in range(NUM_ROW):
            for c in range(NUM_COL):
                # 到达食物
                if (r == 2 and c == 1) :#or (r == c == 1)
                    continue
                nextU[r][c] = max([calculateU(U, r, c, action) for action in range(NUM_ACTIONS)]) # Bellman update
                error = max(error, abs(nextU[r][c]-U[r][c]))
        U = nextU
        printEP(U)
        if error < MAX_ERROR :#* (1-DISCOUNT) / DISCOUNT:
            break
    return U

# 从U得到最优策略
def getOptimalPolicy(U):
    policy = [[-1, -1, -1, -1] for i in range(NUM_ROW)]
    for r in range(NUM_ROW):
        for c in range(NUM_COL):
            # if (r <= 1 and c == 3) :#or (r == c == 1):
            #     continue
            # 选择使效用最大化的行动
            maxAction, maxU = None, -float("inf")
            for action in range(NUM_ACTIONS):
                u = calculateU(U, r, c, action)
                if u > maxU:
                    maxAction, maxU = action, u
            policy[r][c] = maxAction
    return policy



def movement_MDP():
    """
    - State:位置
    - Action:上下左右
    - Reward:体力消耗
    - Discount:r=1
    - 每走一步消耗体力 1,记为-1
    - 找到能量食物结束
    - 最小体力消耗找到能量食物
    """
    global visited_paths, final_path

    # 清空之前的路径记录
    visited_paths = set()
    final_path = []
    
    # 上下左右移动均等，当遇到墙时，相反方向移动的概率更大。

    end = (Map.destination[0],Map.destination[1])
    start = (Map.start[0],Map.start[1])
    discount = DISCOUNT
    global U

    cur_pos = start
    path_list = [start]  # 记录路径
    visited_set = set([start])  # 记录已访问的节点

    print("初始值:\n")
    printEP(U)

    # 值迭代
    U = valueIteration(U)

    # 从U中得到最优策略并打印出来
    policy = getOptimalPolicy(U)
    print("最优策略:\n")
    printEP(policy, True)

    print("RunOutSuccessfly\n")
    # 根据策略移动
    while cur_pos != end:
        # 检查边界，确保不会越界
        if cur_pos[0] < 1 or cur_pos[0] >= Map.height - 1 or cur_pos[1] < 1 or cur_pos[1] >= Map.width - 1:
            break
        
        action = policy[cur_pos[0]-1][cur_pos[1]-1]
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
        if (next_pos[0] >= 0 and next_pos[0] < Map.height and 
            next_pos[1] >= 0 and next_pos[1] < Map.width and
            Map.matrix[next_pos[0]][next_pos[1]] != -1):
            cur_pos = next_pos
            path_list.append(cur_pos)
            visited_set.add(cur_pos)
        else:
            break
    
    final_path = path_list.copy()
    visited_paths = visited_set.copy()
    
    # 显示完整路径
    for p in path_list:
        update_map_search(scene, Map.matrix, Map.path, p, visited=visited_set, final_path_points=final_path)
        time.sleep(0.05)
        QApplication.processEvents()
    
    print("RunOutSuccessfly\n")

    check_reach()

def reward(ax,ay):
    if ax==Map.destination[0] and ay==Map.destination[1] :
        return 100
    elif Map.matrix[ax][ay] == 0 :
        return -1
    elif Map.matrix[ax][ay] == -1 :
        return float("nan") 

def reward(s):
    if s == 7:
        return 100
    else:
        return -1

def statepos(ax,ay):
    return ay-2+3*(ax-2)

def actionchoice(s):
    rand = random.random()
    probs = [states[s][1],states[s][2],states[s][3],states[s][4]]

    for slot, prob in enumerate(probs):
            rand -= prob
            if rand < 0.0:
                return slot    

def transmodel(s,a):
    if a == 0:#UP
        return s-3
    elif a == 1:#DOWN
        return s+3
    elif a == 2:#LEFT
        return s-1
    elif a == 3:#RIGHT
        return s+1

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
        text.setPos(cols * cell_width / 2 - 150, y0 + 10)
        text.setDefaultTextColor(QColor("#525288"))
        scene.addItem(text)
        next_Map_flag = True
        return True

# 事件处理函数已移到 PacmanEnergyfoodWindow 类的 keyPressEvent 方法中

class PacmanEnergyfoodWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        global scene, view, status_label
        
        self.setWindowTitle("PACMAN ENERAGYFOOD")
        menu_height = self.menuBar().height() if self.menuBar() else 25
        status_height = 30
        self.setFixedSize(width, height + menu_height + status_height)
        
        # 创建菜单栏（必须在设置窗口大小之前）
        menubar = self.menuBar()
        try:
            menubar.setNativeMenuBar(False)
        except:
            pass
        filemenu = menubar.addMenu('设置')
        astar_action = filemenu.addAction('A*', movement_astar)
        astar_action.setShortcut('F1')
        filemenu.addSeparator()
        mdp_action = filemenu.addAction('值迭代搜索', movement_MDP)
        mdp_action.setShortcut('F2')
        filemenu.addSeparator()
        restart_action = filemenu.addAction('重新开始', generate_matrix)
        restart_action.setShortcut('F9')
        filemenu.addSeparator()
        exit_action = filemenu.addAction('退出', self.close)
        exit_action.setShortcut('F3')
        
        status_label = QLabel("PAC Man Energyfood")
        self.statusBar().addWidget(status_label)
        
        scene = QGraphicsScene()
        scene.setSceneRect(0, 0, width, height)
        scene.setBackgroundBrush(QBrush(QColor("#F2F2F2")))
        
        view = QGraphicsView(scene, self)
        view.setFixedSize(width, height)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setCentralWidget(view)
        
        global Map, movement_list
        Map = _MapClass(cols, rows)
        movement_list = [Map.start]
        generate_matrix()
    
    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Left or key == Qt.Key.Key_A:
            self._simulate_key_event('Left')
        elif key == Qt.Key.Key_Right or key == Qt.Key.Key_D:
            self._simulate_key_event('Right')
        elif key == Qt.Key.Key_Up or key == Qt.Key.Key_W:
            self._simulate_key_event('Up')
        elif key == Qt.Key.Key_Down or key == Qt.Key.Key_S:
            self._simulate_key_event('Down')
        elif key == Qt.Key.Key_F1:
            movement_astar()
        elif key == Qt.Key.Key_F2:
            movement_MDP()
        elif key == Qt.Key.Key_F3:
            self.close()
        elif key == Qt.Key.Key_F9:
            generate_matrix()
    
    def _simulate_key_event(self, keysym):
        class FakeEvent:
            def __init__(self, keysym):
                self.keysym = keysym
        movement_update_handler(FakeEvent(keysym))

if __name__ == '__main__':
    # 基础参数
    cell_width = 120
    rows = 5
    cols = 5
    height = cell_width * rows
    width = cell_width * cols
    
    click_counter, total_counter, back_counter = 0, 0, 0
    next_Map_flag = False
    scene = None
    view = None
    status_label = None
    Map = None
    movement_list = []
    visited_paths = set()
    final_path = []
    
    t0 = int(time.time())
    t1 = t0
    
    app = QApplication([])
    windows = PacmanEnergyfoodWindow()
    windows.show()
    app.exec()