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
#               0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
brick_matrix=   [  
                [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                [-1, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0,-1, 0, 0,-1, 0, 0, 0, 0, 0, 0,-1, 0, 0,-1],
                [-1, 0, 0,-1, 0,-1, 0, 0, 0, 0,-1, 0, 0, 0, 0,-1, 0,-1, 0, 0,-1, 0, 0,-1,-1,-1, 0,-1, 0, 0,-1],
                [-1, 0, 0,-1, 0,-1, 0, 0, 0, 0,-1,-1,-1,-1, 0,-1, 0,-1, 0, 0,-1, 0, 0, 0, 0, 0, 0,-1, 0, 0,-1],
                [-1, 0, 0,-1, 0,-1, 0, 0, 0, 0,-1, 0, 0,-1, 0,-1, 0,-1, 0, 0, 0, 0, 0,-1, 0, 0, 0,-1, 0, 0,-1],
                [-1, 0, 0, 0, 0,-1, 0, 0, 0, 0,-1, 0, 0, 0, 0,-1, 0,-1, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0,-1],
                [-1, 0, 0, 0, 0,-1, 0, 0, 0, 0,-1, 0, 0, 0, 0,-1, 0, 0, 0, 0,-1,-1, 0,-1, 0, 0, 0, 0, 0, 0,-1],
                [-1, 0, 0,-1,-1,-1,-1,-1, 0, 0,-1, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0,-1,-1,-1,-1,-1],
                [-1, 0,-1,-1, 0, 0, 0,-1, 0, 0,-1, 0,-1, 0, 0,-1, 0, 0,-1, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0,-1],
                [-1, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0,-1, 0, 0,-1, 0, 0,-1, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0,-1],
                [-1, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0,-1, 0, 0,-1, 0, 0,-1, 0, 0, 0, 0,-1, 0, 0,-1, 0, 0, 0,-1],
                [-1, 0, 0,-1, 0, 0, 0,-1, 0, 0, 0, 0,-1, 0, 0,-1, 0, 0,-1, 0, 0, 0, 0,-1, 0, 0,-1, 0, 0, 0,-1],
                [-1, 0, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0,-1, 0, 0,-1, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0,-1],
                [-1, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0,-1],
                [-1, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0,-1, 0, 0,-1, 0, 0, 0,-1],
                [-1, 0,-1, 0, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0,-1],
                [-1, 0,-1, 0, 0,-1, 0,-1, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0,-1],
                [-1, 0,-1,-1, 0,-1, 0,-1, 0,-1,-1,-1, 0,-1, 0,-1, 0,-1,-1, 0,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0,-1],
                [-1, 0, 0,-1,-1,-1, 0, 0, 0,-1,-1,-1, 0,-1, 0,-1, 0, 0,-1, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0,-1],
                [-1, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0,-1, 0,-1, 0, 0,-1, 0, 0, 0, 0, 0,-1,-1, 0,-1,-1,-1,-1],
                [-1, 0, 0,-1, 0, 0, 0,-1,-1,-1,-1,-1, 0,-1, 0,-1, 0, 0,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0,-1],
                [-1, 0, 0,-1,-1,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0, 0, 0, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0,-1],
                [-1, 0, 0, 0, 0,-1, 0,-1, 0, 0, 0,-1, 0,-1, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1],
                [-1, 0, 0,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0, 0, 0,-1, 0, 0,-1, 0, 0, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                [-1,-1,-1,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1],
                [-1, 0, 0, 0, 0,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1],
                [-1, 0, 0, 0, 0, 0, 0,-1, 0,-1, 0, 0, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0,-1],
                [-1, 0, 0,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0,-1],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0,-1,-1,-1,-1,-1,-1,-1],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0,-1],
                [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
                ]
# brick_matrix = brick_matrix.reshape(31,31)

class UnionSet(object):
	"""
	并查集实现，构造函数中的matrix是一个numpy类型
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
    def __init__(self, width = 11, height = 11):
        assert width >= 5 and height >= 5, "Length of width or height must be larger than 5."
        self.width = (width // 2) * 2 + 1
        self.height = (height // 2) * 2 + 1
        self.start = [1, 0]
        self.destination = [self.height - 2, self.width - 1]
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
        self.start = [1, 0]
        self.destination = [self.height - 2, self.width - 1]
        self.generate_matrix(mode, new_matrix)


    def generate_matrix_dfs(self):
        # 地图初始化，并将出口和入口处的值设置为0
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
    根据matrix中每个位置的值绘图：
    -1: 墙壁
    0: 空白
    1: pacman
    2: food
    3: ghost

    """

    canvas.delete("all")
    matrix = copy.copy(matrix)
    
    for p in path:
        matrix[p[0]][p[1]] = 1
    for move in moves:
        matrix[move[0]][move[1]] = 2
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 0:
                draw_cell(canvas, r, c)
            elif matrix[r][c] == -1:
                draw_cell(canvas, r, c, '#525288')
            elif matrix[r][c] == 1:
                draw_cell(canvas, r, c)
                draw_path(canvas, matrix, r, c, '#bc84a8', '#bc84a8')
            elif matrix[r][c] == 2:
                draw_cell(canvas, r, c)
                draw_path(canvas, matrix, r, c, '#ee3f4d', '#ee3f4d')

def update_map(canvas, matrix, path, moves):
    
    canvas.delete("all")
    matrix = copy.copy(matrix)
    for p in path:
        matrix[p[0]][p[1]] = 1
    for move in moves:
        matrix[move[0]][move[1]] = 2

    row, col = movement_list[-1]
    colors = ['#525288', '#F2F2F2', '#525288', '#F2F2F2', '#525288', '#F2F2F2', '#525288', '#F2F2F2']
    
    for r in range(rows):
        for c in range(cols):
            distance = (row - r) * (row - r) + (col - c) * (col - c)
            if distance >= 100:
                color = colors[0:2]
            elif distance >= 60:
                color = colors[2:4]
            elif distance >= 30:
                color = colors[4:6]
            else:
                color = colors[6:8]

            if matrix[r][c] == 0:
                draw_cell(canvas, r, c, color[1])
            elif matrix[r][c] == -1:
                draw_cell(canvas, r, c, color[0])
            elif matrix[r][c] == 1:
                draw_cell(canvas, r, c, color[1])
                draw_path(canvas, matrix, r, c, '#bc84a8', '#bc84a8')
            elif matrix[r][c] == 2:
                draw_cell(canvas, r, c, color[1])
                draw_path(canvas, matrix, r, c, '#ee3f4d', '#ee3f4d')
    

def generate_matrix():
    global movement_list
    global click_counter, back_counter

    click_counter, back_counter = 0, 0
    movement_list = [Map.start]
    Map.generate_matrix(None)
    draw_map(canvas, Map.matrix, Map.path, movement_list)

def movement_update_handler(event):
    global movement_list
    global click_counter, back_counter
    auto_mode = False
    cur_pos = movement_list[-1]
    ops = {'Left': [0, -1], 'Right': [0, 1], 'Up': [-1, 0], 'Down': [1, 0], 'a': [0, -1], 'd': [0, 1], 'w': [-1, 0], 's': [1, 0]}
    r_, c_ = cur_pos[0] + ops[event.keysym][0], cur_pos[1] + ops[event.keysym][1]
    if len(movement_list) > 1 and [r_, c_] == movement_list[-2]:
        click_counter += 1
        back_counter += 1
        movement_list.pop()
        if auto_mode:
            while True:
                cur_pos = movement_list[-1]
                counter = 0
                for d in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                    r_, c_ = cur_pos[0] + d[0], cur_pos[1] + d[1]
                    if c_ >= 0 and Map.matrix[r_][c_] == 0:
                        counter += 1
                if counter != 2:
                    break
                movement_list.pop()
    elif r_ < Map.height and c_ < Map.width and Map.matrix[r_][c_] == 0:
        click_counter += 1
        if auto_mode:
            while True:
                movement_list.append([r_, c_])
                temp_list = []
                for d in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                    r__, c__ = r_ + d[0], c_ + d[1]
                    if c__ < Map.width and Map.matrix[r__][c__] == 0 and [r__, c__] != cur_pos:
                        temp_list.append([r__, c__])
                if len(temp_list) != 1:
                    break
                cur_pos = [r_, c_]
                r_, c_ = temp_list[0]
        else:
            movement_list.append([r_, c_])
    Map.path = []
    update_map(canvas, Map.matrix, Map.path, movement_list)
    check_reach()

def check_reach():
    global next_Map_flag
    if movement_list[-1] == Map.destination:
        print("Congratulations! You reach the goal! Steps used: {}".format(click_counter))
        x0, y0 = cols * cell_width / 2 - 200, 30
        x1, y1 = x0 + 400, y0 + 40
        canvas.create_rectangle(x0, y0, x1, y1, fill = '#F2F2F2', outline ='#525288', width = 3)
        canvas.create_text(cols * cell_width / 2, y0 + 20, text = "Congratulations! You reach the goal! Back steps: {}".format(back_counter), fill = "#525288")
        next_Map_flag = True

def _event_handler(event):
    if event.keysym in ['Left', 'Right', 'Up', 'Down', 'w', 'a', 's', 'd']:
        movement_update_handler(event)
    elif event.keysym == 'F3':
        windows.quit()

if __name__ == '__main__':
    # 基础参数

    cell_width = 20
    rows = 31
    cols = 31
    height = cell_width * rows
    width = cell_width * cols

    click_counter, total_counter, back_counter = 0, 0, 0

    windows = tk.Tk()
    windows.title("PACMAN")
    windows.resizable(0, 0)
    t0 = int(time.time())
    t1 = t0

    #　创建菜单栏
    menubar = tk.Menu(windows)

    filemenu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label='设置', menu=filemenu)
    # filemenu.add_command(label='单 agent', command=_open_map, accelerator='F1')
    # filemenu.add_command(label='双 agent', command=_save_map, accelerator='F2')
    filemenu.add_separator()
    filemenu.add_command(label='退出', command=windows.quit, accelerator='F3')


    windows.config(menu=menubar)
    # end 创建菜单栏

    # 创建状态栏
    label = tk.Label(windows, text="PAC Man", bd=1, anchor='w')  # anchor left align W -- WEST
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