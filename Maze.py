from PyQt6.QtWidgets import (QApplication, QMainWindow, QGraphicsView, 
                              QGraphicsScene, QGraphicsRectItem, QGraphicsTextItem,
                              QLabel, QMenuBar, QMenu, QMessageBox, QFileDialog)
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QColor, QPen, QBrush, QKeyEvent, QMouseEvent
from mazeGenerator import Maze
import pandas as pd
import numpy as np
from PIL import Image
import time
import copy
import math
import os
import seaborn as sns
import matplotlib.pyplot as plt


def draw_cell(scene, row, col, color="#F2F2F2"):
    x0, y0 = col * cell_width, row * cell_width
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


def draw_maze(scene, matrix, path, moves):
    """
    根据matrix中每个位置的值绘图：
    -1: 墙壁
    0: 空白
    1: 参考路径
    2: 移动过的位置
    """
    scene.clear()
    matrix = copy.copy(matrix)
    for p in path:
        matrix[p[0]][p[1]] = 1
    for move in moves:
        matrix[move[0]][move[1]] = 2
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 0:
                draw_cell(scene, r, c)
            elif matrix[r][c] == -1:
                draw_cell(scene, r, c, '#525288')
            elif matrix[r][c] == 1:
                draw_cell(scene, r, c)
                draw_path(scene, matrix, r, c, '#bc84a8', '#bc84a8')
            elif matrix[r][c] == 2:
                draw_cell(scene, r, c)
                draw_path(scene, matrix, r, c, '#ee3f4d', '#ee3f4d')


def update_maze(scene, matrix, path, moves):
    scene.clear()
    matrix = copy.copy(matrix)
    for p in path:
        matrix[p[0]][p[1]] = 1
    for move in moves:
        matrix[move[0]][move[1]] = 2

    row, col = movement_list[-1]
    colors = ['#525288', '#F2F2F2', '#525288', '#F2F2F2', '#525288', '#F2F2F2', '#525288', '#F2F2F2']
    if map_mode > 0:
        colors = ['#232323', '#242424', '#2a2a32', '#424242', '#434368', '#b4b4b4', '#525288', '#F2F2F2']

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
                draw_cell(scene, r, c, color[1])
            elif matrix[r][c] == -1:
                draw_cell(scene, r, c, color[0])
            elif matrix[r][c] == 1:
                draw_cell(scene, r, c, color[1])
                draw_path(scene, matrix, r, c, '#bc84a8', '#bc84a8')
            elif matrix[r][c] == 2:
                draw_cell(scene, r, c, color[1])
                draw_path(scene, matrix, r, c, '#ee3f4d', '#ee3f4d')


def check_reach():
    global next_maze_flag
    if movement_list[-1] == maze.destination:
        print("Congratulations! You reach the goal! Steps used: {}".format(click_counter))
        save_logs(logs_path, set_log_data())
        # 在场景上显示祝贺消息
        x0, y0 = cols * cell_width / 2 - 200, 30
        x1, y1 = x0 + 400, y0 + 40
        rect = QGraphicsRectItem(x0, y0, 400, 40)
        rect.setBrush(QBrush(QColor('#F2F2F2')))
        rect.setPen(QPen(QColor('#525288'), 3))
        scene.addItem(rect)
        text = QGraphicsTextItem("Congratulations! You reach the goal! Back steps: {}".format(back_counter))
        text.setPos(cols * cell_width / 2 - 180, y0 + 10)
        text.setDefaultTextColor(QColor("#525288"))
        scene.addItem(text)
        next_maze_flag = True


def draw_result():
    if len(history_data) == 0:
        QMessageBox.information(None, 'Oppose', '当前没有任何历史数据')
        return
    sns.barplot(x='level', y='value', hue='name', data=history_data)
    plt.title("History score")
    plt.show()


def save_logs(path, text):
    with open(path, 'a+') as file:
        file.write(text)


def movement_update_handler(keysym):
    global movement_list
    global click_counter, back_counter

    cur_pos = movement_list[-1]
    ops = {'Left': [0, -1], 'Right': [0, 1], 'Up': [-1, 0], 'Down': [1, 0], 'a': [0, -1], 'd': [0, 1], 'w': [-1, 0], 's': [1, 0]}
    if keysym not in ops:
        return
    r_, c_ = cur_pos[0] + ops[keysym][0], cur_pos[1] + ops[keysym][1]
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
                    if c_ >= 0 and maze.matrix[r_][c_] == 0:
                        counter += 1
                if counter != 2:
                    break
                movement_list.pop()
    elif r_ < maze.height and c_ < maze.width and maze.matrix[r_][c_] == 0:
        click_counter += 1
        if auto_mode:
            while True:
                movement_list.append([r_, c_])
                temp_list = []
                for d in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                    r__, c__ = r_ + d[0], c_ + d[1]
                    if c__ < maze.width and maze.matrix[r__][c__] == 0 and [r__, c__] != cur_pos:
                        temp_list.append([r__, c__])
                if len(temp_list) != 1:
                    break
                cur_pos = [r_, c_]
                r_, c_ = temp_list[0]
        else:
            movement_list.append([r_, c_])
    maze.path = []
    update_maze(scene, maze.matrix, maze.path, movement_list)
    check_reach()
    set_label_text()


def next_level():
    global click_counter, total_counter, back_counter
    global next_maze_flag
    global level
    global t1

    next_maze_flag = False
    t1 = int(time.time())
    level, total_counter, click_counter, back_counter = level + 1, total_counter + click_counter, 0, 0
    generate_matrix()


def generate_matrix():
    global movement_list
    global map_generate_mode
    global click_counter, back_counter

    if map_size_mode == -1:
        map_generate_mode = 0

    click_counter, back_counter = 0, 0
    movement_list = [maze.start]
    maze.generate_matrix(map_generate_mode, None)
    draw_maze(scene, maze.matrix, maze.path, movement_list)
    set_label_text()


def _set_size():
    QMessageBox.information(None, '上帝', '对不起当前版本暂不支持此功能')


def _set_algo_0():
    global map_generate_mode
    map_generate_mode = 0
    generate_matrix()


def _set_algo_1():
    global map_generate_mode
    map_generate_mode = 1
    generate_matrix()


def _set_algo_2():
    global map_generate_mode
    map_generate_mode = 2
    generate_matrix()


def _set_algo_3():
    global map_generate_mode
    map_generate_mode = 3
    generate_matrix()


def _set_mode_0():
    global map_mode
    map_mode = 0
    update_maze(scene, maze.matrix, maze.path, movement_list)
    set_label_text()


def _set_mode_1():
    global map_mode
    map_mode = 1
    update_maze(scene, maze.matrix, maze.path, movement_list)
    set_label_text()


def _open_map():
    global map_generate_mode

    img_path, _ = QFileDialog.getOpenFileName(None, '打开地图文件', '', 'PNG Files (*.png);;All Files (*)')
    if img_path:
        image = Image.open(img_path)
        matrix = np.asarray(image) / -255
        assert len(matrix) <= 41 and len(matrix[0]) < 91
        map_generate_mode = -1
        _set_size_custom(len(matrix[0]), len(matrix), -1, matrix)


def _save_map():
    path = "{}{}".format(os.getcwd(), image_save_path).replace('\\', '/')
    if not os.path.exists(path):
        os.makedirs(path)
    imgs_len = len(os.listdir(path))
    image = Image.fromarray(-255 * maze.matrix).convert('L')
    image.save("{}map_{}.png".format(path, str(imgs_len)), 'PNG')
    QMessageBox.information(None, '上帝', '当前地图已经保存在{}'.format(path))
    image.show()


def _set_size_custom(width, height, mode, matrix=None):
    global map_size_mode
    global rows, cols
    global movement_list
    global click_counter, back_counter

    click_counter, back_counter = 0, 0
    map_size_mode = mode
    movement_list = [maze.start]
    rows, cols = height, width
    
    # 更新场景大小
    scene.setSceneRect(0, 0, width * cell_width, height * cell_width)
    view.setFixedSize(width * cell_width, height * cell_width)
    windows.setFixedSize(width * cell_width + menu_height, height * cell_width + menu_height + status_height)
    
    if mode == -1:
        maze.resize_matrix(width, height, -1, matrix)
    else:
        maze.resize_matrix(width, height, map_generate_mode, matrix)
    draw_maze(scene, maze.matrix, maze.path, movement_list)
    set_label_text()


def _set_size_31x31():
    _set_size_custom(31, 31, 0)


def _set_size_41x41():
    _set_size_custom(41, 41, 1)


def _set_size_37x81():
    _set_size_custom(81, 37, 2)


def _back_to_start_point():
    global movement_list
    movement_list = [maze.start]
    draw_maze(scene, maze.matrix, maze.path, movement_list)
    set_label_text()


def _set_auto_on():
    global auto_mode
    auto_mode = True


def _set_auto_off():
    global auto_mode
    auto_mode = False


def _developer():
    QMessageBox.information(None, '开发者信息', '当前版本：v 1.0.7\n开发时间：2020年2月\n开发者：Howard Wonanut')


def _man():
    QMessageBox.information(None, '操作说明', '控制移动：方向键\n查看提示：鼠标单击地图中空白处即可查看从起点到点击处的路径(查看一次提示增加20步)\n进入下一关：到达终点后按任意键进入下一关')


def set_label_text():
    message = " Mode: {}   Map size: {}   Algorithm: {}   Total steps: {}   Back steps: {}   Time: {}s".format(
        "Simple" if map_mode == 0 else 'Roguelike',
        ['31x31', '41x41', '81x37'][map_size_mode] if map_size_mode >= 0 else "{}x{}".format(cols, rows),
        ['Kruskal', 'Random DFS', 'Prim', 'Recursive Split'][map_generate_mode] if map_generate_mode >= 0 else 'Unknown',
        click_counter + total_counter, back_counter, int(time.time() - t0))
    status_label.setText(message)
    windows.setWindowTitle("Maze Level-{} Steps-{}".format(level, click_counter))
    return message


def set_log_data():
    return "[{}]Mode:{},Map-size:{},Algorithm:{},Level:{},Steps:{},Back-steps:{},Time-cost:{}\n".format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time()))), "Simple" if map_mode == 0 else 'Roguelike',
        ['31x31', '41x41', '81x37'][map_size_mode] if map_size_mode >= 0 else "{}x{}".format(cols, rows),
        ['Kruskal', 'Random DFS', 'Prim', 'Recursive Split'][map_generate_mode] if map_generate_mode >= 0 else 'Unknown',
        level, click_counter, back_counter, int(time.time() - t1))


class MazeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        global scene, view, status_label, maze, movement_list, windows
        windows = self  # 设置全局变量以便其他函数使用
        
        self.setWindowTitle("Maze")
        menu_height = self.menuBar().height() if self.menuBar() else 25
        status_height = 30
        self.setFixedSize(width + menu_height, height + menu_height + status_height)
        
        # 创建菜单栏
        menubar = self.menuBar()
        try:
            menubar.setNativeMenuBar(False)
        except:
            pass
        
        filemenu = menubar.addMenu('文件')
        open_action = filemenu.addAction('打开地图', _open_map)
        open_action.setShortcut('F1')
        save_action = filemenu.addAction('保存地图', _save_map)
        save_action.setShortcut('F2')
        filemenu.addSeparator()
        exit_action = filemenu.addAction('退出', self.close)
        exit_action.setShortcut('F3')
        
        editmenu = menubar.addMenu('设置')
        back_action = editmenu.addAction('回到起点', _back_to_start_point)
        back_action.setShortcut('F4')
        regenerate_action = editmenu.addAction('换个地图', generate_matrix)
        regenerate_action.setShortcut('F5')
        
        sizemenu = editmenu.addMenu('尺寸设置')
        sizemenu.addAction('31x31', _set_size_31x31)
        sizemenu.addAction('41x41', _set_size_41x41)
        sizemenu.addAction('37x81', _set_size_37x81)
        
        automenu = editmenu.addMenu('自动前进')
        automenu.addAction('开', _set_auto_on)
        automenu.addAction('关', _set_auto_off)
        
        modemenu = editmenu.addMenu('游戏模式')
        modemenu.addAction('普通模式', _set_mode_0)
        modemenu.addAction('迷雾模式', _set_mode_1)
        
        algomenu = editmenu.addMenu('生成算法')
        algomenu.addAction('Kruskal最小生成树算法', _set_algo_0)
        algomenu.addAction('随机深度优先算法', _set_algo_1)
        algomenu.addAction('prim最小生成树算法', _set_algo_2)
        algomenu.addAction('递归分割算法', _set_algo_3)
        
        scoremenu = menubar.addMenu('统计')
        scoremenu.addAction('历史成绩', draw_result)
        
        helpmenu = menubar.addMenu('帮助')
        help_action = helpmenu.addAction('操作说明', _man)
        help_action.setShortcut('F6')
        dev_action = helpmenu.addAction('开发者信息', _developer)
        dev_action.setShortcut('F7')
        
        # 创建场景和视图
        scene = QGraphicsScene()
        scene.setSceneRect(0, 0, width, height)
        scene.setBackgroundBrush(QBrush(QColor("#F2F2F2")))
        
        view = QGraphicsView(scene, self)
        view.setFixedSize(width, height)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view = view  # 保存引用以便在鼠标事件中使用
        view.mousePressEvent = lambda e: self.view_mousePressEvent(e)
        self.setCentralWidget(view)
        
        # 创建状态栏
        status_label = QLabel("Maze Game")
        self.statusBar().addWidget(status_label)
        
        # 初始化游戏
        maze = Maze(cols, rows)
        movement_list = [maze.start]
        generate_matrix()
    
    def keyPressEvent(self, event):
        key = event.key()
        if next_maze_flag:
            next_level()
        elif key == Qt.Key.Key_Left or key == Qt.Key.Key_A:
            movement_update_handler('Left')
        elif key == Qt.Key.Key_Right or key == Qt.Key.Key_D:
            movement_update_handler('Right')
        elif key == Qt.Key.Key_Up or key == Qt.Key.Key_W:
            movement_update_handler('Up')
        elif key == Qt.Key.Key_Down or key == Qt.Key.Key_S:
            movement_update_handler('Down')
        elif key == Qt.Key.Key_F1:
            _open_map()
        elif key == Qt.Key.Key_F2:
            _save_map()
        elif key == Qt.Key.Key_F3:
            self.close()
        elif key == Qt.Key.Key_F4:
            _back_to_start_point()
        elif key == Qt.Key.Key_F5:
            generate_matrix()
        elif key == Qt.Key.Key_F6:
            _man()
        elif key == Qt.Key.Key_F7:
            _developer()
    
    def view_mousePressEvent(self, event):
        # 获取鼠标点击位置相对于视图的坐标
        scene_pos = self.view.mapToScene(event.pos())
        x, y = int(scene_pos.x()), int(scene_pos.y())
        
        row = math.floor(y / cell_width)
        col = math.floor(x / cell_width)
        
        if event.button() == Qt.MouseButton.LeftButton:
            if 0 <= row < rows and 0 <= col < cols and maze.matrix[row][col] == 0:
                maze.find_path_dfs([row, col])
                global click_counter
                click_counter += 20
                update_maze(scene, maze.matrix, maze.path, movement_list)
                set_label_text()
        elif event.button() == Qt.MouseButton.RightButton:
            maze.path = []
            update_maze(scene, maze.matrix, maze.path, movement_list)
            set_label_text()


if __name__ == '__main__':
    # 基础参数
    logs_path = './maze_game.log'
    image_save_path = '/maze_map/'
    cell_width = 20
    rows = 37
    cols = 81
    height = cell_width * rows
    width = cell_width * cols
    level = 1
    click_counter, total_counter, back_counter = 0, 0, 0
    next_maze_flag = False
    history_data = pd.DataFrame(columns=['level', 'name', 'value'])

    # 地图生成算法：0-kruskal，1-dfs，2-prim，3-split
    map_generate_mode = 0
    # 游戏模式：0-简单模式，1-迷雾模式
    map_mode = 0
    # 地图大小：0-31x31, 1-41x41, 2-81x37
    map_size_mode = 1
    # 自动前进模式，默认开
    auto_mode = False

    t0 = int(time.time())
    t1 = t0

    # 全局变量
    scene = None
    view = None
    status_label = None
    maze = None
    movement_list = None
    windows = None  # 将在 MazeWindow.__init__ 中设置
    menu_height = 25
    status_height = 30

    app = QApplication([])
    windows = MazeWindow()
    windows.show()
    app.exec()
