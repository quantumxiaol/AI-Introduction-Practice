import tkinter as tk
from tkinter.messagebox import showinfo
from tkinter import filedialog
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

def draw_pacman(canvas, row, col,color='#B0E0E6'):
    x0, y0 = col * cell_width, row * cell_width
    x1, y1 = x0 + cell_width, y0 + cell_width
    canvas.create_arc(x0, y0, x1, y1,start = 30, extent = 300, fill = color,outline ='yellow',width = 0)

def draw_cell(canvas, row, col, color="#F2F2F2"):
    x0, y0 = col * cell_width, row * cell_width
    x1, y1 = x0 + cell_width, y0 + cell_width
    canvas.create_rectangle(x0, y0, x1, y1, fill = color, outline =color, width = 0)

def draw_food(canvas, row, col, color="#EE3F4D"):
    x0, y0 = col * cell_width, row * cell_width
    x1, y1 = x0 + cell_width, y0 + cell_width
    canvas.create_rectangle(x0, y0, x1, y1, fill = color, outline =color, width = 0)

def draw_path(canvas, matrix, row, col, color, line_color):
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
        canvas.create_rectangle(x0, y0, x1, y1, fill = color, outline = line_color, width = 0)
        x0, y0 = col * cell_width + 2 * cell_width / 5, row * cell_width + 2 * cell_width / 5
        x1, y1 = x0 + cell_width / 5, y0 + 3 * cell_width / 5
    # 右上角
    elif row + 1 < rows and matrix[row][col - 1] >= 1 and matrix[row + 1][col] >= 1:
        x0, y0 = col * cell_width, row * cell_width + 2 * cell_width / 5
        x1, y1 = x0 + 3 * cell_width / 5, y0 + cell_width / 5
        canvas.create_rectangle(x0, y0, x1, y1, fill = color, outline = line_color, width = 0)
        x0, y0 = col * cell_width + 2 * cell_width / 5, row * cell_width + 2 * cell_width / 5
        x1, y1 = x0 + cell_width / 5, y0 + 3 * cell_width / 5
    # 左下角
    elif col + 1 < cols and matrix[row - 1][col] >= 1 and matrix[row][col + 1] >= 1:
        x0, y0 = col * cell_width + 2 * cell_width / 5, row * cell_width
        x1, y1 = x0 + cell_width / 5, y0 + 3 * cell_width / 5
        canvas.create_rectangle(x0, y0, x1, y1, fill = color, outline = line_color, width = 0)
        x0, y0 = col * cell_width + 2 * cell_width / 5, row * cell_width + 2 * cell_width / 5
        x1, y1 = x0 + 3 * cell_width / 5, y0 + cell_width / 5
    # 右下角
    elif matrix[row - 1][col] >= 1 and matrix[row][col - 1] >= 1:
        x0, y0 = col * cell_width, row * cell_width + 2 * cell_width / 5
        x1, y1 = x0 + 3 * cell_width / 5, y0 + cell_width / 5
        canvas.create_rectangle(x0, y0, x1, y1, fill = color, outline = line_color, width = 0)
        x0, y0 = col * cell_width + 2 * cell_width / 5, row * cell_width
        x1, y1 = x0 + cell_width / 5, y0 + 3 * cell_width / 5
    else:
        x0, y0 = col * cell_width + 2 * cell_width / 5, row * cell_width + 2 * cell_width / 5
        x1, y1 = x0 + cell_width / 5, y0 + cell_width / 5
    canvas.create_rectangle(x0, y0, x1, y1, fill = color, outline = line_color, width = 0)

def draw_map(canvas, matrix, path, moves):
    """
    根据matrix中每个位置的值绘图
    -1: 墙壁
    0 : 空白
    1 : pacman
    2 : food
    3 : ghost
    """

    canvas.delete("all")
    matrix = copy.copy(matrix)
    
    # for p in path:
    #     matrix[p[0]][p[1]] = 1
    # for move in moves:
    #     matrix[move[0]][move[1]] = 2
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 0:
                draw_cell(canvas, r, c)
            elif matrix[r][c] == -1:
                draw_cell(canvas, r, c, '#525288')
            elif matrix[r][c] == 1:

                # draw_cell(canvas, r, c)
                draw_pacman(canvas,r,c)
                # draw_path(canvas, matrix, r, c, '#bc84a8', '#bc84a8')
            elif matrix[r][c] == 2:
                draw_food(canvas, r, c)
                # draw_path(canvas, matrix, r, c, '#ee3f4d', '#ee3f4d')

def update_map(canvas, matrix, path, moves):
    
    canvas.delete("all")
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
                draw_cell(canvas, r, c, colors[1])
            elif matrix[r][c] == -1:
                draw_cell(canvas, r, c, colors[0])
            elif matrix[r][c] == 1:
                draw_pacman(canvas,r,c)
                # draw_cell(canvas, r, c, colors[1])
                # draw_path(canvas, matrix, r, c, '#bc84a8', '#bc84a8')
            elif matrix[r][c] == 2:
                draw_food(canvas, r, c)
                # draw_cell(canvas, r, c, colors[1])
                # draw_path(canvas, matrix, r, c, '#ee3f4d', '#ee3f4d')
    

def generate_matrix():
    global movement_list
    global click_counter, back_counter

    click_counter, back_counter = 0, 0
    movement_list = [Map.start]
    Map.generate_matrix(None)
    draw_map(canvas, Map.matrix, Map.path, movement_list)

def update_map_search(canvas, matrix, path, moves):
    
    canvas.delete("all")
    matrix = copy.copy(matrix)
    # for p in path:
    #     matrix[p[0]][p[1]] = 1
    matrix[moves[0]][moves[1]] = -3

    row, col = movement_list[-1]
    colors = ['#525288', '#F2F2F2', '#525288', '#F2F2F2', '#525288', '#F2F2F2', '#525288', '#F2F2F2']
    
    for r in range(rows):
        for c in range(cols):

            if matrix[r][c] == 0:
                draw_cell(canvas, r, c, colors[1])
            elif matrix[r][c] == -1:
                draw_cell(canvas, r, c, colors[0])
            elif matrix[r][c] == -2:
                draw_cell(canvas, r, c, colors[1])#"#525266")                
            elif matrix[r][c] == 1:
                draw_pacman(canvas,r,c)
            elif matrix[r][c] == 2:
                draw_food(canvas, r, c)
            elif matrix[r][c] == -3:
                draw_cell(canvas, r, c, "#525266")
    # windows.after(500,update_map_search(canvas, matrix, path, moves))
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

    update_map(canvas, Map.matrix, Map.path, movement_list)
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
    lab=copy.copy(Map.matrix)
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

    while True:
        # get first state (least cost)
        state = fringe.pop(0)
        # print(state)
        # goal check
        (i, j) = state[0]
        # print(state[0])
        
        if (i, j) == end:
            path = [state[0]] + state[1]
            path.reverse()
            print(path)
            print("RunOutSuccessfly\n")
            for p in path:
                update_map_search(canvas, Map.matrix, Map.path, (p[0],p[1]))
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

def movement_a():
    global movement_list
    global click_counter, back_counter

    cur_pos = movement_list[-1]

    update_map(canvas, Map.matrix, Map.path, movement_list)
    check_reach()

# 给定状态的Agent行为概率分布，RV
# （状态 上移概率、下移概率、左移概率、右移概率）
state = [   (0,0,0.5,0,0.5),(1,0,0.4,0.3,0.3),(2,0,0.5,0.5,0),
            (3,0.3,0.3,0,0.4),(4,0.25,0.25,0.25,0.25),(5,0.3,0.3,0.4,0),
            (6,0.5,0,0,0.5),(7,0.4,0,0.3,0.3),(8,0.5,0,0.5,0)]

# 上下左右
pacman_action = [0,1,2,3]

action = []

statehistroy = []

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
    # 上下左右移动均等，当遇到墙时，相反方向移动的概率更大。

    end = (Map.destination[0],Map.destination[1])
    start = (Map.start[0],Map.start[1])
    discount = 1

    statehistroy.append(state(start))
    value=[-1,-1,-1,-1,-1,-1,-1,100,-1]

    # step1
    



    # print("state=%d,postion=(%d,%d),reward=%f",)

    check_reach()

def reward(ax,ay):
    if ax==Map.destination[0] and ay==Map.destination[1] :
        return 100
    elif Map.matrix[ax][ay] == 0 :
        return -1
    elif Map.matrix[ax][ay] == -1 :
        return float("nan") 

def statepos(ax,ay):
    return ay-2+3(ax-2)

def actionchoice(s):
    rand = random.random()
    probs = [state[s][1],state[s][2],state[s][3],state[s][4]]

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
        canvas.create_rectangle(x0, y0, x1, y1, fill = '#F2F2F2', outline ='#525288', width = 3)
        canvas.create_text(cols * cell_width / 2, y0 + 20, text = "Congratulations! You Eat all the food! ", fill = "#525288")
        next_Map_flag = True
        return True

def _event_handler(event):
    if event.keysym in ['Left', 'Right', 'Up', 'Down', 'w', 'a', 's', 'd']:
        movement_update_handler(event)
    elif event.keysym == 'F1':
        movement_astar()
    elif event.keysym == 'F2':
        movement_MDP()
    elif event.keysym == 'F3':
        windows.quit()

if __name__ == '__main__':
    # 基础参数

    cell_width = 120
    rows = 5
    cols = 5
    height = cell_width * rows
    width = cell_width * cols

    click_counter, total_counter, back_counter = 0, 0, 0

    windows = tk.Tk()
    windows.title("PACMAN ENERAGYFOOD")
    windows.resizable(0, 0)
    t0 = int(time.time())
    t1 = t0

    #　创建菜单栏
    menubar = tk.Menu(windows)

    filemenu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label='设置', menu=filemenu)
    filemenu.add_command(label='A*', command=movement_astar, accelerator='F1')
    filemenu.add_separator()
    filemenu.add_command(label='值迭代搜索', command=movement_MDP, accelerator='F2')    
    filemenu.add_separator()
    filemenu.add_command(label='退出', command=windows.quit, accelerator='F3')

    windows.config(menu=menubar)
    # end 创建菜单栏

    # 创建状态栏
    label = tk.Label(windows, text="PAC Man Energyfood", bd=1, anchor='w')  # anchor left align W -- WEST
    label.pack(side="bottom", fill='x')

    canvas = tk.Canvas(windows, background="#F2F2F2", width = width, height = height)
    canvas.pack()

    Map = Map(cols, rows)
    movement_list = [Map.start]
    # Map.print_matrix()
    generate_matrix()
    
    # canvas.bind("<Button-1>", _paint_answer_path)
    # canvas.bind("<Button-3>", _reset_answer_path)
    canvas.bind_all("<KeyPress>", _event_handler)

    windows.mainloop()