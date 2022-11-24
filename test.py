import tkinter as tk

import tkinter as tk
from tkinter.messagebox import showinfo
from tkinter import filedialog
from mazeGenerator import Maze
import pandas as pd
import numpy as np
from PIL import Image
import time
import copy
import math
import os

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
                [-1, 0,-1,-1, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1, 2, 0, 0, 0,-1, 0,-1,-1, 0,-1, 0,-1,-1, 0,-1, 0,-1],
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
                [-1,-1, 0,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0,-1,-1,-1,-1, 3,-1,-1],
                [-1,-1, 0, 0, 0, 0, 0,-1,-1,-1,-1, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0,-1,-1,-1,-1,-1,-1,-1],
                [-1,-1, 0,-1,-1,-1, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0,-1],
                [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
                ]


Labyrinth =[[0, 0, 0, 0, 0, 0, 1, 0],
[0, 1, 0, 1, 1, 1, 1, 0],
[0, 1, 1, 1, 0, 1, 0, 0],
[0, 1, 0, 0, 0, 0, 0, 0],
[0, 1, 1, 0, 1, 1, 3, 0],
[0, 0, 1, 1, 1, 0, 0, 0],
[0, 1, 2, 0, 1, 1, 1, 0],
[0, 1, 0, 0, 0, 0, 0, 0]]

L =[[-1, -1, -1, -1, -1, -1, 0, -1],
[-1, 0, -1, 0, 0, 0, 0, -1],
[-1, 0, 2, 0, -1, 0, -1, -1],
[-1, 0, -1, -1, -1, -1, -1, -1],
[-1, 0, 0, -1, 0, 0, 3, -1],
[-1, -1, 0, 0, 0, -1, -1, -1],
[-1, 0, 0, -1, 0, 0, 0, -1],
[-1, 0, -1, -1, -1, -1, -1, -1]]


def astar(lab):
    # first, let's look for the beginning position, there is better but it works
    (i_s, j_s) = [[(i, j) for j, cell in enumerate(row) if cell == 2] for i, row in enumerate(lab) if 2 in row][0][0]
    # and take the goal position (used in the heuristic)
    (i_e, j_e) = [[(i, j) for j, cell in enumerate(row) if cell == 3] for i, row in enumerate(lab) if 3 in row][0][0]

    width = len(lab[0])
    height = len(lab)

    heuristic = lambda i, j: abs(i_e - i) + abs(j_e - j)
    comp = lambda state: state[2] + state[3] # get the total cost

    # small variation for easier code, state is (coord_tuple, previous, path_cost, heuristic_cost)
    fringe = [((i_s, j_s), list(), 0, heuristic(i_s, j_s))]
    visited = {} # empty set
    print(fringe)
    # maybe limit to prevent too long search
    while True:

        # get first state (least cost)
        state = fringe.pop(0)
        print(state,"\n")
        # time.sleep(1)
        # goal check
        (i, j) = state[0]
        if lab[i][j] == 3:
            path = [state[0]] + state[1]
            path.reverse()
            print(path)
            # print(fringe)
            return path

        # set the cost (path is enough since the heuristic won't change)
        visited[(i, j)] = state[2] 

        # explore neighbor
        neighbor = list()
        if i > 0 and lab[i-1][j] >= 0: #top
            neighbor.append((i-1, j))
        if i < height and lab[i+1][j] >= 0:
            neighbor.append((i+1, j))
        if j > 0 and lab[i][j-1] >= 0:
            neighbor.append((i, j-1))
        if j < width and lab[i][j+1] >= 0:
            neighbor.append((i, j+1))

        for n in neighbor:
            next_cost = state[2] + 1
            if n in visited and visited[n] >= next_cost:
                continue
            fringe.append((n, [state[0]] + state[1], next_cost, heuristic(n[0], n[1])))

        # resort the list (SHOULD use a priority queue here to avoid re-sorting all the time)
        fringe.sort(key=comp)

astar(brick_matrix)

# 调用Tk()创建主窗口
root_window =tk.Tk()
root_window.title('Pac Man')
window_width = 1200
window_height = 900
root_window.minsize(window_width,window_height)

screenwidth = root_window.winfo_screenwidth()
screenheight = root_window.winfo_screenheight()
size_geo = '%dx%d+%d+%d' % (window_width, window_height, (screenwidth-window_width)/2, (screenheight-window_height)/2)
root_window.geometry(size_geo)
canvas = tk.Canvas(root_window,width = 1200,height = 900,bg='white')
x0,y0,x1,y1 = 10,10,100,100
man = canvas.create_arc(x0, y0, x1, y1,start = 30, extent = 300, fill = '#B0E0E6',outline ='yellow',width = 2)
canvas.pack()
# # 添加按钮
# button = tk.Button(root_window,text="关闭",command=root_window.quit)
# # 这里将按钮放置在主窗口的底部
# button.pack(side="bottom")


#开启主循环，让窗口处于显示状态
root_window.mainloop()
