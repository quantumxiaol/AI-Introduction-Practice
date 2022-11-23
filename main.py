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
