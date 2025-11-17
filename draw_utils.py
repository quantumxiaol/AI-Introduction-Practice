"""
绘制工具函数 - 提供地图绘制相关功能
"""
from PyQt6.QtWidgets import QGraphicsRectItem, QGraphicsEllipseItem
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPen, QBrush
import copy


def draw_pacman(scene, row, col, cell_width, color='#B0E0E6'):
    """绘制吃豆人"""
    x0, y0 = col * cell_width, row * cell_width
    ellipse = QGraphicsEllipseItem(x0, y0, cell_width, cell_width)
    ellipse.setStartAngle(30 * 16)  # PyQt6 使用 1/16 度为单位
    ellipse.setSpanAngle(300 * 16)
    ellipse.setBrush(QBrush(QColor(color)))
    ellipse.setPen(QPen(QColor('yellow'), 0))
    scene.addItem(ellipse)


def draw_cell(scene, row, col, cell_width, color="#F2F2F2"):
    """绘制单元格"""
    x0, y0 = col * cell_width, row * cell_width
    rect = QGraphicsRectItem(x0, y0, cell_width, cell_width)
    rect.setBrush(QBrush(QColor(color)))
    rect.setPen(QPen(QColor(color), 0))
    scene.addItem(rect)


def draw_food(scene, row, col, cell_width, color="#EE3F4D"):
    """绘制食物"""
    x0, y0 = col * cell_width, row * cell_width
    rect = QGraphicsRectItem(x0, y0, cell_width, cell_width)
    rect.setBrush(QBrush(QColor(color)))
    rect.setPen(QPen(QColor(color), 0))
    scene.addItem(rect)


def draw_path(scene, matrix, row, col, cell_width, rows, cols, color, line_color):
    """绘制路径"""
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


def draw_map(scene, matrix, path, moves, cell_width, rows, cols):
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
    
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 0:
                draw_cell(scene, r, c, cell_width)
            elif matrix[r][c] == -1:
                draw_cell(scene, r, c, cell_width, '#525288')
            elif matrix[r][c] == 1:
                draw_pacman(scene, r, c, cell_width)
            elif matrix[r][c] == 2:
                draw_food(scene, r, c, cell_width)


def update_map(scene, matrix, path, moves, cell_width, rows, cols, movement_list):
    """更新地图显示"""
    scene.clear()
    matrix = copy.copy(matrix)
    
    row, col = movement_list[-1]
    colors = ['#525288', '#F2F2F2', '#525288', '#F2F2F2', '#525288', '#F2F2F2', '#525288', '#F2F2F2']
    
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 0:
                draw_cell(scene, r, c, cell_width, colors[1])
            elif matrix[r][c] == -1:
                draw_cell(scene, r, c, cell_width, colors[0])
            elif matrix[r][c] == 1:
                draw_pacman(scene, r, c, cell_width)
            elif matrix[r][c] == 2:
                draw_food(scene, r, c, cell_width)


def update_map_search(scene, matrix, path, moves, cell_width, rows, cols, map_obj, visited=None, final_path_points=None):
    """更新地图显示，高亮当前搜索位置和路径"""
    scene.clear()
    display_matrix = copy.deepcopy(matrix)
    
    # 标记已访问的路径（用浅蓝色）
    if visited:
        for pos in visited:
            if isinstance(pos, tuple) and len(pos) == 2:
                r, c = pos
                if (r, c) != (map_obj.start[0], map_obj.start[1]) and (r, c) != (map_obj.destination[0], map_obj.destination[1]):
                    display_matrix[r][c] = -4  # 已访问标记
    
    # 标记最终路径（用绿色）
    if final_path_points:
        for pos in final_path_points:
            if isinstance(pos, tuple) and len(pos) == 2:
                r, c = pos
                if (r, c) != (map_obj.start[0], map_obj.start[1]) and (r, c) != (map_obj.destination[0], map_obj.destination[1]):
                    display_matrix[r][c] = -5  # 最终路径标记
    
    # 标记当前搜索位置
    if isinstance(moves, tuple) and len(moves) == 2:
        display_matrix[moves[0]][moves[1]] = -3
    
    # 显示起点和终点（确保它们始终显示）
    display_matrix[map_obj.start[0]][map_obj.start[1]] = 1
    display_matrix[map_obj.destination[0]][map_obj.destination[1]] = 2

    colors = ['#525288', '#F2F2F2', '#525288', '#F2F2F2', '#525288', '#F2F2F2', '#525288', '#F2F2F2']
    
    for r in range(rows):
        for c in range(cols):
            if display_matrix[r][c] == 0:
                draw_cell(scene, r, c, cell_width, colors[1])
            elif display_matrix[r][c] == -1:
                draw_cell(scene, r, c, cell_width, colors[0])
            elif display_matrix[r][c] == -2:
                draw_cell(scene, r, c, cell_width, "#CCCCCC")  # 已访问过的路径用浅灰色
            elif display_matrix[r][c] == -4:
                draw_cell(scene, r, c, cell_width, "#ADD8E6")  # 已访问路径用浅蓝色
            elif display_matrix[r][c] == -5:
                draw_cell(scene, r, c, cell_width, "#90EE90")  # 最终路径用浅绿色
            elif display_matrix[r][c] == 1:
                draw_pacman(scene, r, c, cell_width)
            elif display_matrix[r][c] == 2:
                draw_food(scene, r, c, cell_width)
            elif display_matrix[r][c] == -3:
                draw_cell(scene, r, c, cell_width, "#FFD700")  # 当前搜索位置用金色高亮

