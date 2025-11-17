"""
吃豆人能量食物游戏主程序
"""
from PyQt6.QtWidgets import (QApplication, QMainWindow, QGraphicsView, 
                              QGraphicsScene, QGraphicsRectItem, QGraphicsTextItem,
                              QLabel, QMenuBar, QMenu)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPen, QBrush
import time

from map_utils import PacmanMap
from draw_utils import draw_map, update_map, update_map_search
from search_algorithms import movement_astar
from mdp_algorithm import movement_mdp


def generate_matrix():
    """生成地图矩阵"""
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
        text.setPos(cols * cell_width / 2 - 150, y0 + 10)
        text.setDefaultTextColor(QColor("#525288"))
        scene.addItem(text)
        next_Map_flag = True
        return True


class PacmanEnergyfoodWindow(QMainWindow):
    """吃豆人能量食物游戏窗口"""
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
        astar_action = filemenu.addAction('A*', lambda: movement_astar(scene, Map, cell_width, rows, cols))
        astar_action.setShortcut('F1')
        filemenu.addSeparator()
        mdp_action = filemenu.addAction('值迭代搜索', lambda: movement_mdp(scene, Map, cell_width, rows, cols))
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
        Map = PacmanMap(cols, rows)
        movement_list = [Map.start]
        generate_matrix()
    
    def keyPressEvent(self, event):
        """处理键盘事件"""
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
            movement_astar(scene, Map, cell_width, rows, cols)
        elif key == Qt.Key.Key_F2:
            movement_mdp(scene, Map, cell_width, rows, cols)
        elif key == Qt.Key.Key_F3:
            self.close()
        elif key == Qt.Key.Key_F9:
            generate_matrix()
    
    def _simulate_key_event(self, keysym):
        """模拟键盘事件"""
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
