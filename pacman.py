"""
吃豆人游戏主程序
"""
from PyQt6.QtWidgets import (QApplication, QMainWindow, QGraphicsView, 
                              QGraphicsScene, QGraphicsRectItem, QGraphicsTextItem,
                              QLabel, QMenuBar, QMenu)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPen, QBrush
import time
import copy

from map_utils import PacmanMap
from draw_utils import draw_map, update_map, update_map_search
from search_algorithms import movement_dfs, movement_bfs, movement_ucs, movement_astar


def generate_matrix(mode=None):
    """生成地图矩阵"""
    global movement_list
    global click_counter, back_counter
    global visited_paths, final_path
    global map_generate_mode

    click_counter, back_counter = 0, 0
    movement_list = [Map.start]  # 重置移动列表
    Map.path = []  # 清空路径
    visited_paths = set()  # 清空已访问路径
    final_path = []  # 清空最终路径
    
    # 如果指定了模式，更新全局模式
    if mode is not None:
        map_generate_mode = mode
    
    # 重新生成地图
    Map.generate_matrix(map_generate_mode)
    # 确保起点和终点正确设置（覆盖任何之前的修改）
    Map.matrix[Map.start[0]][Map.start[1]] = 1
    Map.matrix[Map.destination[0]][Map.destination[1]] = 2
    # 清理搜索标记（-2, -3, -4, -5等）
    for r in range(Map.height):
        for c in range(Map.width):
            if Map.matrix[r][c] in [-2, -3, -4, -5]:
                # 恢复被标记为已访问或搜索中的位置
                if (r, c) != (Map.start[0], Map.start[1]) and (r, c) != (Map.destination[0], Map.destination[1]):
                    # 根据地图模式决定恢复方式
                    if map_generate_mode == 'brick':
                        # 预设地图模式，使用原始砖块地图
                        from map_utils import BRICK_MATRIX_31, BRICK_MATRIX_5
                        if Map.height == 5:
                            original_matrix = BRICK_MATRIX_5
                        else:
                            original_matrix = BRICK_MATRIX_31
                        if original_matrix[r][c] == -1:
                            Map.matrix[r][c] = -1
                        else:
                            Map.matrix[r][c] = 0
                    else:
                        # 随机地图模式，恢复为空地
                        Map.matrix[r][c] = 0
    # 再次确保起点和终点正确
    Map.matrix[Map.start[0]][Map.start[1]] = 1
    Map.matrix[Map.destination[0]][Map.destination[1]] = 2
    draw_map(scene, Map.matrix, Map.path, movement_list, cell_width, rows, cols)


def movement_update_handler(event):
    """处理移动更新"""
    global movement_list
    global click_counter, back_counter

    cur_pos = movement_list[-1]

    ops = {'Left': [0, -1], 'Right': [0, 1], 'Up': [-1, 0], 'Down': [1, 0], 
           'a': [0, -1], 'd': [0, 1], 'w': [-1, 0], 's': [1, 0]}
    r_, c_ = cur_pos[0] + ops[event.keysym][0], cur_pos[1] + ops[event.keysym][1]
    if len(movement_list) > 1 and [r_, c_] == movement_list[-2]:
        click_counter += 1
        back_counter += 1
        movement_list.pop()
        if r_ < Map.height and c_ < Map.width and Map.matrix[r_][c_] == 0:
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
    elif Map.matrix[cur_pos[0]+ops[event.keysym][0]][cur_pos[1]+ops[event.keysym][1]] == -1:
        Map.matrix[cur_pos[0]][cur_pos[1]] = 1

    update_map(scene, Map.matrix, Map.path, movement_list, cell_width, rows, cols, movement_list)
    check_reach()


def check_reach():
    """检查是否到达终点"""
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
    """吃豆人游戏窗口"""
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
        dfs_action = filemenu.addAction('深度优先', lambda: movement_dfs(scene, Map, cell_width, rows, cols))
        dfs_action.setShortcut('F1')
        bfs_action = filemenu.addAction('广度优先', lambda: movement_bfs(scene, Map, cell_width, rows, cols))
        bfs_action.setShortcut('F2')
        ucs_action = filemenu.addAction('一致代价', lambda: movement_ucs(scene, Map, cell_width, rows, cols))
        ucs_action.setShortcut('F3')
        astar_action = filemenu.addAction('A*', lambda: movement_astar(scene, Map, cell_width, rows, cols))
        astar_action.setShortcut('F4')
        filemenu.addSeparator()
        
        # 地图生成菜单
        mapmenu = filemenu.addMenu('地图生成')
        mapmenu.addAction('预设地图', lambda: generate_matrix('brick'))
        mapmenu.addAction('DFS算法', lambda: generate_matrix('dfs'))
        mapmenu.addAction('Prim算法', lambda: generate_matrix('prim'))
        mapmenu.addAction('Kruskal算法', lambda: generate_matrix('kruskal'))
        mapmenu.addAction('递归分割', lambda: generate_matrix('split'))
        
        filemenu.addSeparator()
        exit_action = filemenu.addAction('退出', self.close)
        exit_action.setShortcut('F5')
        restart_action = filemenu.addAction('重新开始', lambda: generate_matrix())
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
        Map = PacmanMap(cols, rows)
        movement_list = [Map.start]
        generate_matrix()
        
    def keyPressEvent(self, event):
        """处理键盘事件"""
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
            movement_dfs(scene, Map, cell_width, rows, cols)
        elif key == Qt.Key.Key_F2:
            movement_bfs(scene, Map, cell_width, rows, cols)
        elif key == Qt.Key.Key_F3:
            movement_ucs(scene, Map, cell_width, rows, cols)
        elif key == Qt.Key.Key_F4:
            movement_astar(scene, Map, cell_width, rows, cols)
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
    visited_paths = set()
    final_path = []
    map_generate_mode = 'brick'  # 默认使用预设地图
    
    t0 = int(time.time())
    t1 = t0

    app = QApplication([])
    windows = PacmanWindow()
    windows.show()
    app.exec()
