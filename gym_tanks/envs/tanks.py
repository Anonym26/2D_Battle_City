# Импорт необходимых библиотек
import numpy as np
import os, pygame, time, random, uuid, sys
import matplotlib.pyplot as plt
import multiprocessing
import queue
from queue import Empty

# Импорт библиотек для создания и тренировки ИИ-бота
import heapq
import math

# Импорт библиотек для работы с Gym
import gymnasium

from skimage.transform import rescale
from collections import deque


'''
========================================================================================================================
                                                            НАСТРОЙКИ
========================================================================================================================
'''


# ITU-R 601-2 luma transform
def rgb_to_grayscale(rgb_array):
    """
        Преобразование цветного изображения в оттенки серого с использованием трансформации ITU-R 601-2.

        Параметры:
            rgb_array (numpy.ndarray): Входной массив RGB изображения.

        Возвращает:
            numpy.ndarray: Массив в оттенках серого, уменьшенный вдвое по размеру.
    """
    # Определяем веса для каналов RGB
    weights = np.array([0.2989, 0.5870, 0.1140])
    # Вычисляем значение серого для каждого пикселя
    grayscale_array = np.dot(rgb_array[..., :3], weights)
    # Убираем информационную панель и уменьшаем изображение вдвое
    grayscale_array_noinfobar = grayscale_array[:, :416]
    grayscale_array_downscaled = rescale(grayscale_array_noinfobar, 1/2.0, anti_aliasing=True, mode='reflect')
    # Округляем значения и преобразуем в uint8
    grayscale_array_rounded = np.round(grayscale_array_downscaled).astype(np.uint8)

    return grayscale_array_rounded


def Vmanhattan_distance(a, b):
    """
        Вычисление Манхеттенского расстояния между двумя точками.

        Параметры:
            a (tuple): Координаты первой точки (x1, y1).
            b (tuple): Координаты второй точки (x2, y2).

        Возвращает:
            int: Манхеттенское расстояние между точками.
    """
    x1, y1 = a
    x2, y2 = b
    return abs(x1 - x2) + abs(y1 - y2)


def Veuclidean_distance(a, b):
    """
        Вычисление Евклидова расстояния между двумя точками.

        Параметры:
            a (tuple): Координаты первой точки (x1, y1).
            b (tuple): Координаты второй точки (x2, y2).

        Возвращает:
            float: Евклидово расстояние между точками.
    """
    x1, y1 = a
    x2, y2 = b
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def Vinline_with_enemy(player_rect, enemy_rect):
    """
        Определение, находится ли враг на одной линии с игроком.

        Параметры:
            player_rect (pygame.Rect): Прямоугольник игрока.
            enemy_rect (pygame.Rect): Прямоугольник врага.

        Возвращает:
            int: Код положения врага относительно игрока:
                0 - Враг сверху.
                2 - Враг снизу.
                3 - Враг слева.
                1 - Враг справа.
                4 - Враг не находится на одной линии.
    """
    # Проверка вертикального выравнивания
    if enemy_rect.left <= player_rect.centerx <= enemy_rect.right and abs(player_rect.top - enemy_rect.bottom) <= 151:
        # Враг сверху
        if enemy_rect.bottom <= player_rect.top:
            return 0
        # Враг снизу
        elif player_rect.bottom <= enemy_rect.top:
            return 2
    # Проверка горизонтального выравнивания
    if enemy_rect.top <= player_rect.centery <= enemy_rect.bottom and abs(player_rect.left - enemy_rect.right) <= 151:
        # Враг слева
        if enemy_rect.right <= player_rect.left:
            return 3
        # Враг справа
        elif player_rect.right <= enemy_rect.left:
            return 1
    # Враг не на одной линии
    return 4


def Vbullet_avoidance(player_rect_out, bullet_info_list):  # bullet_info_list = bullets?
    """
        Определение направления уклонения от ближайшей пули.

        Параметры:
            player_rect_out (pygame.Rect): Прямоугольник игрока.
            bullet_info_list (list): Список информации о пулях, содержащий прямоугольник пули и её направление.

        Возвращает:
            int: Код направления уклонения:
                0 - Уклонение вверх.
                2 - Уклонение вниз.
                3 - Уклонение влево.
                1 - Уклонение вправо.
                4 - Уклонение не требуется.
    """
    obs_bullet_avoidance_direction = 4

    player_rect = player_rect_out

    # Сортируем пули по Евклидову расстоянию до игрока
    sorted_bullet_info_list = sorted(bullet_info_list, key=lambda x: Veuclidean_distance((x[0].left, x[0].top), (player_rect.centerx, player_rect.centery)))

    # Устанавливаем минимальное расстояние до снаряда как бесконечность
    if sorted_bullet_info_list:
        min_dist_with_bullet = Veuclidean_distance((sorted_bullet_info_list[0][0].left, sorted_bullet_info_list[0][0].top), (player_rect.centerx, player_rect.centery))
    else:
        min_dist_with_bullet = float(1e30000)

    # Активируем уклонение, если расстояние до снаряда <= 120
    if min_dist_with_bullet <= 120:
        # Выбираем ближайший снаряд
        bullet_rect = sorted_bullet_info_list[0][0]
        bullet_direction = sorted_bullet_info_list[0][1]
        # Если расстояние по x <= 25
        if abs(bullet_rect.centerx - player_rect.centerx) <= 25:
            # Если расстояние по x <= 5
            if abs(bullet_rect.centerx - player_rect.centerx) <= 5:
                # Снаряд движется вверх и находится снизу игрока
                if bullet_direction == 0 and bullet_rect.top > player_rect.top:
                    obs_bullet_avoidance_direction = 2
                # Пуля движется вниз и находится сверху игрока
                if bullet_direction == 2 and bullet_rect.top < player_rect.top:
                    obs_bullet_avoidance_direction = 0
        # Если расстояние по y <= 25
        elif abs(bullet_rect.centery - player_rect.centery) <= 25:
            # Если расстояние по y <= 5
            if abs(bullet_rect.centery - player_rect.centery) <= 5:
                # Сняряд движется вправо и находится слева от игрока
                if bullet_direction == 1 and bullet_rect.left < player_rect.left:
                    obs_bullet_avoidance_direction = 3
                # Сняряд движется влево и находится справа от игрока
                if bullet_direction == 3 and bullet_rect.left > player_rect.left:
                    obs_bullet_avoidance_direction = 1

    return obs_bullet_avoidance_direction

def antiStupidBlock(player_direction, player_rect, base_rect):
    """
    Определение необходимости блокировки движения игрока в сторону базы.

    Параметры:
        player_direction (int): Направление движения игрока (0 - вверх, 1 - вправо, 2 - вниз, 3 - влево).
        player_rect (pygame.Rect): Прямоугольник игрока.
        base_rect (pygame.Rect): Прямоугольник базы.

    Возвращает:
        int: 1 - Блокировать движение, 0 - Разрешить движение.
    """
    # Проверка вертикального выравнивания
    if base_rect.left <= player_rect.centerx <= base_rect.right:
        # База сверху и игрок движется вверх
        if base_rect.bottom <= player_rect.top and player_direction == 0:
            return 1
        # База снизу и игрок движется вниз
        elif player_rect.bottom <= base_rect.top and player_direction == 2:
            return 1
    # Проверка горизонтального выравнивания
    if base_rect.top <= player_rect.centery <= base_rect.bottom:
        # База слева и игрок движется влево
        if base_rect.right <= player_rect.left and player_direction == 3:
            return 1
        # База справа и игрок движется вправо
        elif player_rect.right <= base_rect.left and player_direction == 1:
            return 1
    return 0


'''
===============================================================================================================================
                                                            AI АГЕНТ
===============================================================================================================================
'''

class PriorityQueue:
    """
    Класс для управления приоритетной очередью, используемой в алгоритмах поиска пути.

    Методы:
        empty() -> bool: Проверяет, является ли очередь пустой.
        put(item, priority): Добавляет элемент в очередь с указанным приоритетом.
        get() -> any: Извлекает элемент с наивысшим приоритетом.
    """
    def __init__(self):
        """Инициализация пустой очереди."""
        self.elements = []

    def empty(self):
        """Проверяет, пуста ли очередь."""
        return len(self.elements) == 0

    def put(self, item, priority):
        """Добавляет элемент в очередь с указанным приоритетом.

        Параметры:
            item (any): Элемент для добавления.
            priority (int): Приоритет элемента.
        """
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        """Извлекает элемент с наивысшим приоритетом.

        Возвращает:
            any: Элемент с наивысшим приоритетом.
        """
        return heapq.heappop(self.elements)[1]


class ai_agent():
    """
    Класс агента ИИ для управления стратегией игры.

    Атрибуты:
        mapinfo (list): Информация о карте.
        castle_rect (pygame.Rect): Координаты замка.

    Методы:
        __init__(): Инициализация агента.
        operations(p_mapinfo, c_control): Основной цикл работы агента.
    """
    mapinfo = []
    # Координаты замка
    castle_rect = pygame.Rect(12 * 16, 24 * 16, 32, 32)

    def __init__(self):
        """Инициализация объекта агента."""
        self.mapinfo = []

    def operations(self, p_mapinfo, c_control):
        """
        Основной метод, описывающий логику управления агентом.

        Параметры:
            p_mapinfo (list): Информация о текущем состоянии карты.
            c_control (object): Объект управления игроком.
        """
        while True:
            self.Get_mapInfo(p_mapinfo)  # Обновляем информацию о карте
            player_rect = self.mapinfo[3][0][0]  # Прямоугольник игрока

            # Сортируем врагов по Манхэттенскому расстоянию до замка
            sorted_enemy_with_distance_to_castle = sorted(
                self.mapinfo[1],
                key=lambda x: self.manhattan_distance(x[0].center, self.castle_rect.center))
            # Сортируем врагов по Манхэттенскому расстоянию до игрока
            sorted_enemy_with_distance_to_player = sorted(
                self.mapinfo[1],
                key=lambda x: self.manhattan_distance(x[0].center, player_rect.center))

            # Стандартное положение игрока
            default_pos_rect = pygame.Rect(195, 3, 26, 26)
            if sorted_enemy_with_distance_to_castle:
                # Преследуем врага, если он близко к замку
                if self.manhattan_distance(sorted_enemy_with_distance_to_castle[0][0].topleft, self.castle_rect.topleft) < 150:
                    enemy_rect = sorted_enemy_with_distance_to_castle[0][0]
                    enemy_direction = sorted_enemy_with_distance_to_castle[0][1]
                else:  # В противном случае преследуем ближайшего к игроку врага
                    enemy_rect = sorted_enemy_with_distance_to_player[0][0]
                    enemy_direction = sorted_enemy_with_distance_to_player[0][1]

                # Проверяем, находится ли игрок на линии с врагом
                inline_direction = self.inline_with_enemy(player_rect, enemy_rect)
                # Вычисляем следующее действие с использованием алгоритма A*
                astar_direction = self.a_star(player_rect, enemy_rect, 6)
                # Определяем необходимость стрельбы или уклонения от пуль
                shoot, direction = self.bullet_avoidance(self.mapinfo[3][0], 6, self.mapinfo[0], astar_direction, inline_direction)
                # Обновляем стратегию в зависимости от вычисленных данных
                self.Update_Strategy(c_control, shoot, direction)
                time.sleep(0.005)
            else:
                # Перемещаемся в стандартное положение, если врагов рядом нет
                astar_direction = self.a_star(player_rect, default_pos_rect, 6)
                if astar_direction is not None:
                    self.Update_Strategy(c_control, 0, astar_direction)
                else:
                    self.Update_Strategy(c_control, 0, 0)


            # ------------------------------------------------------------------------------------------------------

    # Функция для получения текущей информации о карте
    def Get_mapInfo(self, p_mapinfo):
        """
        Обновляет информацию о текущей карте из очереди.

        Параметры:
            p_mapinfo (queue.Queue): Очередь с данными карты.
        """
        # Проверяем, что очередь не пуста
        if p_mapinfo.empty() != True:
            try:
                # Получаем данные из очереди
                self.mapinfo = p_mapinfo.get(False)
            except queue.empty:
                # Игнорируем, если очередь пуста
                skip_this = True

    # Функция для обновления стратегии ИИ
    def Update_Strategy(self, c_control, shoot, move_dir):
        """
        Обновляет стратегию ИИ (стрельба и движение) в очереди управления.

        Параметры:
            c_control (queue.Queue): Очередь управления.
            shoot (bool): Флаг необходимости стрельбы.
            move_dir (int): Направление движения.
        """
        # Обновляем стратегию, только если очередь управления пуста
        if c_control.empty() == True:
            c_control.put([shoot, move_dir])

    # Проверка необходимости стрельбы игрока на основе выравнивания с врагом
    def should_fire(self, player_rect, enemy_rect_info_list):
        """
        Проверяет, должен ли игрок стрелять на основе выравнивания с врагом.

        Параметры:
            player_rect (pygame.Rect): Прямоугольник игрока.
            enemy_rect_info_list (list): Список информации о врагах (их прямоугольники и направления).

        Возвращает:
            bool: True, если нужно стрелять, иначе False.
        """
        for enemy_rect_info in enemy_rect_info_list:
            # Проверяем, находится ли враг на одной линии с игроком
            if self.inline_with_enemy(player_rect, enemy_rect_info[0]) is not False:
                return True

    # Реализация алгоритма A* для поиска пути к цели
    def a_star(self, start_rect, goal_rect, speed):
        """
        Реализация алгоритма A* для нахождения пути к цели.

        Параметры:
            start_rect (pygame.Rect): Начальное положение.
            goal_rect (pygame.Rect): Целевое положение.
            speed (int): Скорость передвижения.

        Возвращает:
            int: Код направления движения (0 - вверх, 1 - вправо, 2 - вниз, 3 - влево).
        """
        # Преобразуем прямоугольники в координаты сетки
        start = (start_rect.left, start_rect.top)
        goal = (goal_rect.left, goal_rect.top)

        # Инициализируем очередь с приоритетом и вспомогательные структуры данных
        frontier = PriorityQueue()
        came_from = {}
        cost_so_far = {}

        # Добавляем стартовую позицию в очередь
        frontier.put(start, 0)
        came_from[start] = None
        cost_so_far[start] = 0

        while not frontier.empty():
            # Получаем текущую позицию
            current_left, current_top = frontier.get()
            current = (current_left, current_top)

            # Проверяем, достигнута ли цель
            temp_rect = pygame.Rect(current_left, current_top, 26, 26)
            if self.is_goal(temp_rect, goal_rect):
                break

            # Исследуем соседние узлы
            for next in self.find_neighbour(current_top, current_left, speed, goal_rect):
                # Вычисляем новую стоимость
                new_cost = cost_so_far[current] + speed
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(goal, next)
                    frontier.put(next, priority)
                    came_from[next] = current

        next = None
        dir_cmd = None
        while current != start:
            # Восстанавливаем путь от цели к старту
            next = current
            current = came_from[current]

        if next:
            # Определяем направление движения
            next_left, next_top = next
            current_left, current_top = current
            if current_top > next_top:
                dir_cmd = 0  # Вверх
            elif current_top < next_top:
                dir_cmd = 2  # Вниз
            elif current_left > next_left:
                dir_cmd = 3  # Влево
            elif current_left < next_left:
                dir_cmd = 1  # Вправо
        return dir_cmd

    # Вычисление Манхэттенского расстояния между двумя точками
    def manhattan_distance(self, a, b):
        """
        Вычисляет Манхэттенское расстояние между двумя точками.

        Параметры:
            a (tuple): Координаты первой точки (x1, y1).
            b (tuple): Координаты второй точки (x2, y2).

        Возвращает:
            int: Манхэттенское расстояние.
        """
        x1, y1 = a
        x2, y2 = b
        return abs(x1 - x2) + abs(y1 - y2)

    # Вычисление Евклидова расстояния между двумя точками
    def euclidean_distance(self, a, b):
        """
        Вычисляет Евклидово расстояние между двумя точками.

        Параметры:
            a (tuple): Координаты первой точки (x1, y1).
            b (tuple): Координаты второй точки (x2, y2).

        Возвращает:
            float: Евклидово расстояние.
        """
        x1, y1 = a
        x2, y2 = b
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # Эвристическая функция для алгоритма A*
    def heuristic(self, a, b):
        """
        Эвристическая функция для алгоритма A* на основе Манхэттенского расстояния.

        Параметры:
            a (tuple): Координаты первой точки (x1, y1).
            b (tuple): Координаты второй точки (x2, y2).

        Возвращает:
            int: Эвристическое значение.
        """
        return self.manhattan_distance(a, b)

    # Проверка, находятся ли два прямоугольника в целевой позиции
    def is_goal(self, rect1, rect2):
        """
        Проверяет, находятся ли два прямоугольника в целевой позиции.

        Параметры:
            rect1 (pygame.Rect): Первый прямоугольник.
            rect2 (pygame.Rect): Второй прямоугольник.

        Возвращает:
            bool: True, если целевая позиция достигнута, иначе False.
        """
        center_x1, center_y1 = rect1.center
        center_x2, center_y2 = rect2.center
        return abs(center_x1 - center_x2) <= 7 and abs(center_y1 - center_y2) <= 7

    def find_neighbour(self, top, left, speed, goal_rect):
        """
        Определяет доступные для движения соседние позиции, избегая столкновений.

        Параметры:
            top (int): Верхняя координата текущей позиции.
            left (int): Левая координата текущей позиции.
            speed (int): Скорость движения.
            goal_rect (pygame.Rect): Прямоугольник цели.

        Возвращает:
            list: Список допустимых позиций для движения (координаты).
        """
        global obs_flag_top_occupied, obs_flag_bottom_occupied, obs_flag_left_occupied, obs_flag_right_occupied
        obs_flag_top_occupied = False
        obs_flag_bottom_occupied = False
        obs_flag_left_occupied = False
        obs_flag_right_occupied = False

        # Прямоугольник (левая координата, верхняя координата, ширина, высота)
        allowable_move = []

        # Движение вверх
        new_top = top - speed
        new_left = left
        if not (new_top < 0):  # Проверяем, находится ли новая позиция в пределах поля
            move_up = True
            temp_rect = pygame.Rect(new_left, new_top, 26, 26)

            # Проверяем столкновения с врагами, кроме цели
            for enemy in self.mapinfo[1]:
                if enemy[0] is not goal_rect:
                    if temp_rect.colliderect(enemy[0]):
                        move_up = False
                        break

            # Проверяем столкновения с объектами, исключая траву
            if move_up:
                for tile in self.mapinfo[2]:
                    if tile[1] != 4:
                        if temp_rect.colliderect(tile[0]):
                            move_up = False
                            break

            if move_up:
                allowable_move.append((new_left, new_top))

        # Движение вправо
        new_top = top
        new_left = left + speed
        if not (new_left > (416 - 26)):  # Проверяем, находится ли новая позиция в пределах поля
            move_right = True
            temp_rect = pygame.Rect(new_left, new_top, 26, 26)

            # Проверяем столкновения с врагами, кроме цели
            for enemy in self.mapinfo[1]:
                if enemy[0] is not goal_rect:
                    if temp_rect.colliderect(enemy[0]):
                        move_right = False
                        break

            # Проверяем столкновения с объектами, исключая траву
            if move_right:
                for tile in self.mapinfo[2]:
                    if tile[1] != 4:
                        if temp_rect.colliderect(tile[0]):
                            move_right = False
                            break

            if move_right:
                allowable_move.append((new_left, new_top))

        # Движение вниз
        new_top = top + speed
        new_left = left
        if not (new_top > (416 - 26)):  # Проверяем, находится ли новая позиция в пределах поля
            move_down = True
            temp_rect = pygame.Rect(new_left, new_top, 26, 26)

            # Проверяем столкновения с врагами, кроме цели
            for enemy in self.mapinfo[1]:
                if enemy[0] is not goal_rect:
                    if temp_rect.colliderect(enemy[0]):
                        move_down = False
                        break

            # Проверяем столкновения с объектами, исключая траву
            if move_down:
                for tile in self.mapinfo[2]:
                    if tile[1] != 4:
                        if temp_rect.colliderect(tile[0]):
                            move_down = False
                            break
            if move_down:
                allowable_move.append((new_left, new_top))

        # Движение влево
        new_top = top
        new_left = left - speed
        if not (new_left < 0):  # Проверяем, находится ли новая позиция в пределах поля
            move_left = True
            temp_rect = pygame.Rect(new_left, new_top, 26, 26)

            # Проверяем столкновения с врагами, кроме цели
            for enemy in self.mapinfo[1]:
                if enemy[0] is not goal_rect:
                    if temp_rect.colliderect(enemy[0]):
                        move_left = False
                        break

            # Проверяем столкновения с объектами, исключая траву
            if move_left:
                for tile in self.mapinfo[2]:
                    if tile[1] != 4:
                        if temp_rect.colliderect(tile[0]):
                            move_left = False
                            break

            if move_left:
                allowable_move.append((new_left, new_top))

        return allowable_move


    def inline_with_enemy(self, player_rect, enemy_rect):
        """
        Проверяет, находится ли игрок на одной линии с врагом.

        Параметры:
            player_rect (pygame.Rect): Прямоугольник игрока.
            enemy_rect (pygame.Rect): Прямоугольник врага.

        Возвращает:
            int: Направление врага относительно игрока (0 - сверху, 1 - справа, 2 - снизу, 3 - слева).
            bool: False, если игрок не находится на одной линии с врагом.
        """
        # Проверяем горизонтальное выравнивание игрока с врагом в заданном диапазоне
        if enemy_rect.left <= player_rect.centerx <= enemy_rect.right and abs(player_rect.top - enemy_rect.bottom) <= 151:
            if enemy_rect.bottom <= player_rect.top:
                return 0  # Враг сверху
            elif player_rect.bottom <= enemy_rect.top:
                return 2  # Враг снизу
        # Проверяем вертикальное выравнивание игрока с врагом в заданном диапазоне
        if enemy_rect.top <= player_rect.centery <= enemy_rect.bottom and abs(player_rect.left - enemy_rect.right) <= 151:
            if enemy_rect.right <= player_rect.left:
                return 3  # Враг слева
            elif player_rect.right <= enemy_rect.left:
                return 1  # Враг справа
        return False

    def bullet_avoidance(self, player_info, speed, bullet_info_list, direction_from_astar, inlined_with_enemy):
        """
        Рассчитывает направление уклонения от пули и определяет необходимость стрельбы.

        Параметры:
            player_info (tuple): Информация об игроке (прямоугольник и данные).
            speed (int): Скорость игрока.
            bullet_info_list (list): Список информации о пулях.
            direction_from_astar (int): Направление движения, рассчитанное алгоритмом A*.
            inlined_with_enemy (int): Направление выравнивания с врагом.

        Возвращает:
            tuple: Флаг стрельбы (1 - стрелять, 0 - не стрелять) и направление движения.
        """
        directions = []  # Список возможных направлений движения
        player_rect = player_info[0]  # Получаем прямоугольник игрока

        # Сортируем список пуль по Евклидову расстоянию до игрока
        sorted_bullet_info_list = sorted(bullet_info_list, key=lambda x: self.euclidean_distance((x[0].left, x[0].top), (player_rect.centerx, player_rect.centery)))

        shoot = 0  # Инициализируем флаг стрельбы
        min_dist_with_bullet = float(1e30000)  # Устанавливаем минимальное расстояние как очень большое значение

        if sorted_bullet_info_list:
            # Вычисляем минимальное расстояние до ближайшей пули
            min_dist_with_bullet = self.euclidean_distance((sorted_bullet_info_list[0][0].left, sorted_bullet_info_list[0][0].top), (player_rect.centerx, player_rect.centery))

        if min_dist_with_bullet <= 120:
            # Получаем данные о ближайшей пуле
            bullet_rect = sorted_bullet_info_list[0][0]
            bullet_direction = sorted_bullet_info_list[0][1]

            if abs(bullet_rect.centerx - player_rect.centerx) <= 30:
                if abs(bullet_rect.centerx - player_rect.centerx) <= 5:
                    if bullet_direction == 0 and bullet_rect.top > player_rect.top:
                        directions.append(2)  # Уклонение вниз
                        shoot = 1
                    if bullet_direction == 2 and bullet_rect.top < player_rect.top:
                        directions.append(0)  # Уклонение вверх
                        shoot = 1
                else:
                    if bullet_rect.left > player_rect.centerx:
                        directions.append(3)  # Уклонение влево
                    else:
                        directions.append(1)  # Уклонение вправо

            elif abs(bullet_rect.centery - player_rect.centery) <= 30:
                if abs(bullet_rect.centery - player_rect.centery) <= 5:
                    if bullet_direction == 1 and bullet_rect.left < player_rect.left:
                        directions.append(3)  # Уклонение влево
                        shoot = 1
                    if bullet_direction == 3 and bullet_rect.left > player_rect.left:
                        directions.append(1)  # Уклонение вправо
                        shoot = 1
                else:
                    if bullet_rect.top > player_rect.centery:
                        directions.append(0)  # Уклонение вверх
                        directions.append(2)  # Уклонение вниз
                    else:
                        directions.append(2)  # Уклонение вниз
                        directions.append(0)  # Уклонение вверх

            else:
                if inlined_with_enemy == direction_from_astar:
                    shoot = 1
                directions.append(direction_from_astar)

                if bullet_direction == 0 or bullet_direction == 2:
                    if bullet_rect.left > player_rect.left:
                        if 1 in directions:
                            directions.remove(1)
                        else:
                            if 3 in directions:
                                directions.remove(3)

                if bullet_direction == 1 or bullet_direction == 3:
                    if bullet_rect.top > player_rect.top:
                        if 2 in directions:
                            directions.remove(2)
                    else:
                        if 0 in directions:
                            directions.remove(0)
        else:
            if inlined_with_enemy == direction_from_astar:
                shoot = 1
            directions.append(direction_from_astar)

        if directions:
            for direction in directions:
                new_left, new_top = self.calculate_new_position(player_rect, direction, speed)
                temp_rect = pygame.Rect(new_left, new_top, 26, 26)

                if self.is_valid_position(new_top, new_left):
                    if not self.is_collision(temp_rect):
                        if not self.will_hit_base_or_obstacles(player_rect, direction):
                            return shoot, direction
                else:
                    opposite_direction = self.get_opposite_direction(direction)
                    new_left, new_top = self.calculate_new_position(player_rect, opposite_direction, speed)
                    temp_rect = pygame.Rect(new_left, new_top, 26, 26)
                    if self.is_valid_position(new_top, new_left) and not self.is_collision(temp_rect):
                        if not self.will_hit_base_or_obstacles(player_rect, opposite_direction):
                            return shoot, opposite_direction
        else:
            return shoot, 4

        return shoot, direction_from_astar

    def calculate_new_position(self, player_rect, direction, speed):
        """
        Вычисляет новую позицию игрока на основе направления и скорости.

        Параметры:
            player_rect (pygame.Rect): Прямоугольник игрока.
            direction (int): Направление движения (0 - вверх, 1 - вправо, 2 - вниз, 3 - влево).
            speed (int): Скорость передвижения.

        Возвращает:
            tuple: Новые координаты (new_left, new_top).
        """
        if direction == 0:  # Вверх
            new_left = player_rect.left
            new_top = player_rect.top - speed
        elif direction == 1:  # Вправо
            new_left = player_rect.left + speed
            new_top = player_rect.top
        elif direction == 2:  # Вниз
            new_left = player_rect.left
            new_top = player_rect.top + speed
        elif direction == 3:  # Влево
            new_left = player_rect.left - speed
            new_top = player_rect.top
        else:  # Без изменений
            new_top = player_rect.top
            new_left = player_rect.left
        return new_left, new_top

    def is_valid_position(self, top, left):
        """
        Проверяет, находится ли позиция в пределах карты.

        Параметры:
            top (int): Верхняя координата позиции.
            left (int): Левая координата позиции.

        Возвращает:
            bool: True, если позиция валидна, иначе False.
        """
        return 0 <= top <= 416 - 26 and 0 <= left <= 416 - 26

    def is_collision(self, temp_rect):
        """
        Проверяет столкновение с объектами карты.

        Параметры:
            temp_rect (pygame.Rect): Прямоугольник для проверки столкновений.

        Возвращает:
            bool: True, если есть столкновение, иначе False.
        """
        for tile_info in self.mapinfo[2]:
            tile_rect = tile_info[0]
            tile_type = tile_info[1]
            if tile_type != 4:  # Исключаем определённый тип объектов (например, препятствия)
                if temp_rect.colliderect(tile_rect):
                    return True
        return False

    def get_opposite_direction(self, direction):
        """
        Возвращает противоположное направление движения.

        Параметры:
            direction (int): Текущее направление (0 - вверх, 1 - вправо, 2 - вниз, 3 - влево).

        Возвращает:
            int: Противоположное направление.
        """
        return (direction + 2) % 4

    def will_hit_base_or_obstacles(self, player_rect, direction):
        """
        Проверяет, попадёт ли выстрел в базу или препятствия.

        Параметры:
            player_rect (pygame.Rect): Прямоугольник игрока.
            direction (int): Направление выстрела (0 - вверх, 1 - вправо, 2 - вниз, 3 - влево).

        Возвращает:
            bool: True, если выстрел попадёт в базу или препятствия, иначе False.
        """
        bullet_path = self.simulate_bullet_path(player_rect, direction)
        for obstacle in self.mapinfo[2]:  # Проверяем все препятствия на карте
            if bullet_path.colliderect(obstacle[0]):
                return True  # Выстрел попадёт в препятствие
        if bullet_path.colliderect(self.castle_rect):
            return True  # Выстрел попадёт в базу
        return False

    def simulate_bullet_path(self, player_rect, direction):
        """
        Симулирует траекторию выстрела игрока.

        Параметры:
            player_rect (pygame.Rect): Прямоугольник игрока.
            direction (int): Направление выстрела (0 - вверх, 1 - вправо, 2 - вниз, 3 - влево).

        Возвращает:
            pygame.Rect: Прямоугольник, представляющий траекторию выстрела.
        """
        if direction in [0, 2]:  # Вертикальный выстрел (вверх или вниз)
            return pygame.Rect(player_rect.centerx - 2, 0, 4, 416)
        if direction in [1, 3]:  # Горизонтальный выстрел (вправо или влево)
            return pygame.Rect(0, player_rect.centery - 2, 416, 4)
        return pygame.Rect(0, 0, 0, 0)  # Нет выстрела


'''
========================================================================================================================
                                                    ТАНКОВЫЕ СРАЖЕНИЯ
========================================================================================================================
'''


class myRect(pygame.Rect):
    """
    Класс myRect, расширяющий pygame.Rect для добавления свойства type.

    Атрибуты:
        type (any): Тип объекта, связанный с прямоугольником.
    """
    def __init__(self, left, top, width, height, type):
        super().__init__(left, top, width, height)
        self.type = type

class Timer(object):
    """
    Класс Timer для управления таймерами.

    Методы:
        add(interval, f, repeat=-1): Добавляет новый таймер.
        destroy(uuid_nr): Удаляет таймер по его UUID.
        update(time_passed): Обновляет все таймеры и вызывает их колбэки.
    """
    def __init__(self):
        """Инициализация объекта Timer."""
        self.timers = []

    def add(self, interval, f, repeat=-1):
        """
        Добавляет новый таймер.

        Параметры:
            interval (int): Интервал вызова таймера в миллисекундах.
            f (function): Функция, вызываемая таймером.
            repeat (int): Количество повторений (-1 для бесконечного повторения).

        Возвращает:
            uuid.UUID: Уникальный идентификатор таймера.
        """
        options = {
            "interval": interval,
            "callback": f,
            "repeat": repeat,
            "times": 0,
            "time": 0,
            "uuid": uuid.uuid4()
        }
        self.timers.append(options)
        return options["uuid"]

    def destroy(self, uuid_nr):
        """
        Удаляет таймер по его UUID.

        Параметры:
            uuid_nr (uuid.UUID): Уникальный идентификатор таймера.
        """
        for timer in self.timers:
            if timer["uuid"] == uuid_nr:
                self.timers.remove(timer)
                return

    def update(self, time_passed):
        """
        Обновляет таймеры и вызывает их функции.

        Параметры:
            time_passed (int): Время, прошедшее с последнего вызова.
        """
        for timer in self.timers:
            timer["time"] += time_passed
            if timer["time"] > timer["interval"]:
                timer["time"] -= timer["interval"]
                timer["times"] += 1
                if timer["repeat"] > -1 and timer["times"] == timer["repeat"]:
                    self.timers.remove(timer)
                try:
                    timer["callback"]()
                except Exception as e:
                    try:
                        self.timers.remove(timer)
                    except:
                        pass

class Castle():
    """
    Класс Castle, представляющий замок игрока.

    Атрибуты:
        STATE_STANDING (int): Состояние "Стоящий".
        STATE_DESTROYED (int): Состояние "Разрушен".
        STATE_EXPLODING (int): Состояние "Взрыв".
        rect (pygame.Rect): Прямоугольник, представляющий замок.
        image (pygame.Surface): Текущая текстура замка.
        active (bool): Состояние активности замка.
    """
    (STATE_STANDING, STATE_DESTROYED, STATE_EXPLODING) = range(3)

    def __init__(self):
        """Инициализация объекта Castle."""
        global sprites

        # Загрузка изображений замка
        self.img_undamaged = sprites.subsurface(0, 15*2, 16*2, 16*2)
        self.img_destroyed = sprites.subsurface(16*2, 15*2, 16*2, 16*2)

        # Начальная позиция замка
        self.rect = pygame.Rect(12*16, 24*16, 32, 32)

        # Инициализация замка в целостном состоянии
        self.rebuild()

    def draw(self):
        """
        Отрисовывает замок на экране.
        """
        global screen

        screen.blit(self.image, self.rect.topleft)

        if self.state == self.STATE_EXPLODING:
            if not self.explosion.active:
                self.state = self.STATE_DESTROYED
                del self.explosion
            else:
                self.explosion.draw()

    def rebuild(self):
        """
        Восстанавливает замок в исходное состояние.
        """
        self.state = self.STATE_STANDING
        self.image = self.img_undamaged
        self.active = True

    def destroy(self):
        """
        Разрушает замок, активируя анимацию взрыва.
        """
        self.state = self.STATE_EXPLODING
        self.explosion = Explosion(self.rect.topleft)
        self.image = self.img_destroyed
        self.active = False


class Bullet():
    """
        Класс Bullet представляет снаряд, выпущенный игроком или врагом.

        Атрибуты:
            DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT: Константы направлений движения.
            STATE_REMOVED, STATE_ACTIVE, STATE_EXPLODING: Константы состояния снаряда.
            OWNER_PLAYER, OWNER_ENEMY: Константы, определяющие владельца снаряда.
    """
    # Константы направлений
    (DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT) = range(4)

    # Константы состояний
    (STATE_REMOVED, STATE_ACTIVE, STATE_EXPLODING) = range(3)

    # Константы владельцев
    (OWNER_PLAYER, OWNER_ENEMY) = range(2)

    def __init__(self, level, position, direction, damage=100, speed=5):
        """
        Инициализация снаряда.

        Параметры:
            level (object): Уровень, на котором выпущен снаряд.
            position (tuple): Начальная позиция снаряда.
            direction (int): Направление движения снаряда.
            damage (int): Урон от снаряда.
            speed (int): Скорость движения снаряда.
        """
        global sprites

        self.level = level
        self.direction = direction
        self.damage = damage
        self.owner = None
        self.owner_class = None

        # Тип мощности снаряда (1 - обычный, 2 - пробивает сталь)
        self.power = 1

        # Загрузка изображения снаряда
        self.image = sprites.subsurface(75*2, 74*2, 3*2, 4*2)

        # Установка позиции и вращение изображения в зависимости от направления
        if direction == self.DIR_UP:
            self.rect = pygame.Rect(position[0] + 11, position[1] - 8, 6, 8)
        elif direction == self.DIR_RIGHT:
            self.image = pygame.transform.rotate(self.image, 270)
            self.rect = pygame.Rect(position[0] + 26, position[1] + 11, 8, 6)
        elif direction == self.DIR_DOWN:
            self.image = pygame.transform.rotate(self.image, 180)
            self.rect = pygame.Rect(position[0] + 11, position[1] + 26, 6, 8)
        elif direction == self.DIR_LEFT:
            self.image = pygame.transform.rotate(self.image, 90)
            self.rect = pygame.Rect(position[0] - 8, position[1] + 11, 8, 6)

        # Изображения взрыва
        self.explosion_images = [
            sprites.subsurface(0, 80*2, 32*2, 32*2),
            sprites.subsurface(32*2, 80*2, 32*2, 32*2),
        ]

        self.speed = speed
        self.state = self.STATE_ACTIVE

    def draw(self):
        """
        Отрисовывает снаряд на экране.
        """
        global screen
        if self.state == self.STATE_ACTIVE:
            screen.blit(self.image, self.rect.topleft)
        elif self.state == self.STATE_EXPLODING:
            self.explosion.draw()


    def update(self):
        """
        Обновляет положение снаряда и проверяет столкновения.
        """
        global castle, players, enemies, bullets

        if self.state == self.STATE_EXPLODING:
            if not self.explosion.active:
                self.destroy()
                del self.explosion

        if self.state != self.STATE_ACTIVE:
            return

        # Перемещение снаряда
        if self.direction == self.DIR_UP:
            self.rect.topleft = [self.rect.left, self.rect.top - self.speed]
            if self.rect.top < 0:
                if play_sounds and self.owner == self.OWNER_PLAYER:
                    sounds["steel"].play()
                self.explode()
                return
        elif self.direction == self.DIR_RIGHT:
            self.rect.topleft = [self.rect.left + self.speed, self.rect.top]
            if self.rect.left > (416 - self.rect.width):
                if play_sounds and self.owner == self.OWNER_PLAYER:
                    sounds["steel"].play()
                self.explode()
                return
        elif self.direction == self.DIR_DOWN:
            self.rect.topleft = [self.rect.left, self.rect.top + self.speed]
            if self.rect.top > (416 - self.rect.height):
                if play_sounds and self.owner == self.OWNER_PLAYER:
                    sounds["steel"].play()
                self.explode()
                return
        elif self.direction == self.DIR_LEFT:
            self.rect.topleft = [self.rect.left - self.speed, self.rect.top]
            if self.rect.left < 0:
                if play_sounds and self.owner == self.OWNER_PLAYER:
                    sounds["steel"].play()
                self.explode()
                return

        has_collided = False

        # Проверяем столкновения с препятствиями
        rects = self.level.obstacle_rects
        collisions = self.rect.collidelistall(rects)
        if collisions != []:
            for i in collisions:
                if self.level.hitTile(rects[i].topleft, self.power, self.owner == self.OWNER_PLAYER):
                    has_collided = True
        if has_collided:
            self.explode()
            return

        # Проверяем столкновения с другими снарядами
        for bullet in bullets:
            if self.state == self.STATE_ACTIVE and bullet.owner != self.owner and bullet != self and self.rect.colliderect(
                    bullet.rect):
                self.destroy()
                self.explode()
                return

        # Проверяем столкновения с игроками
        for player in players:
            if player.state == player.STATE_ALIVE and self.rect.colliderect(player.rect):
                if player.bulletImpact(self.owner == self.OWNER_PLAYER, self.damage, self.owner_class):
                    if self.owner == self.OWNER_ENEMY:
                        self.destroy()
                    return

        # Проверяем столкновения с врагами
        for enemy in enemies:
            if enemy.state == enemy.STATE_ALIVE and self.rect.colliderect(enemy.rect):
                if enemy.bulletImpact(self.owner == self.OWNER_ENEMY, self.damage, self.owner_class):
                    self.destroy()
                    return

        # Проверяем столкновения с замком
        if castle.active and self.rect.colliderect(castle.rect):
            castle.destroy()
            self.destroy()
            return

        # Проверяем столкновения с врагами
        for enemy in enemies:
            if enemy.state == enemy.STATE_ALIVE and self.rect.colliderect(enemy.rect):
                if enemy.bulletImpact(self.owner == self.OWNER_ENEMY, self.damage, self.owner_class):
                    self.destroy()
                    return

        # Проверяем столкновения с замком
        if castle.active and self.rect.colliderect(castle.rect):
            castle.destroy()
            self.destroy()
            return

    def explode(self):
        """
        Запускает взрыв снаряда.
        """
        global screen
        if self.state != self.STATE_REMOVED:
            self.state = self.STATE_EXPLODING
            self.explosion = Explosion([self.rect.left - 13, self.rect.top - 13], None, self.explosion_images)

    def destroy(self):
        """
        Удаляет снаряд с карты.
        """
        self.state = self.STATE_REMOVED


class Label():
    """
    Класс Label представляет текстовую метку на экране.

    Атрибуты:
        position (tuple): Позиция метки на экране.
        text (str): Текст метки.
        active (bool): Состояние активности метки.
        font (pygame.Font): Шрифт для текста метки.
    """
    def __init__(self, position, text="", duration=None):
        """
        Инициализация текстовой метки.

        Параметры:
            position (tuple): Позиция метки на экране.
            text (str): Текст метки.
            duration (int, optional): Продолжительность отображения метки.
        """
        self.position = position
        self.active = True
        self.text = text
        self.font = pygame.font.SysFont("Arial", 13)

        if duration is not None:
            gtimer.add(duration, lambda: self.destroy(), 1)

    def draw(self):
        """
        Отображает текстовую метку на экране.
        """
        pass  # Реализация отображения отключена в предоставленном коде

    def destroy(self):
        """
        Удаляет метку с экрана.
        """
        self.active = False


class Explosion():
    """
    Класс Explosion представляет анимацию взрыва.

    Атрибуты:
        position (list): Координаты взрыва.
        active (bool): Состояние активности взрыва.
        images (list): Список изображений для анимации взрыва.
    """
    def __init__(self, position, interval=None, images=None):
        """
        Инициализация анимации взрыва.

        Параметры:
            position (tuple): Позиция взрыва.
            interval (int, optional): Интервал между кадрами анимации.
            images (list, optional): Список изображений для анимации взрыва.
        """
        global sprites

        self.position = [position[0] - 16, position[1] - 16]
        self.active = True

        if interval is None:
            interval = 1

        if images is None:
            images = [
                sprites.subsurface(0, 80 * 2, 32 * 2, 32 * 2),
                sprites.subsurface(32 * 2, 80 * 2, 32 * 2, 32 * 2),
                sprites.subsurface(64 * 2, 80 * 2, 32 * 2, 32 * 2)
            ]

        images.reverse()

        self.images = images.copy()
        self.image = self.images.pop()

        gtimer.add(interval, lambda: self.update(), len(self.images) + 1)

    def draw(self):
        """
        Отображает текущий кадр анимации взрыва.
        """
        global screen
        screen.blit(self.image, self.position)

    def update(self):
        """
        Переходит к следующему кадру анимации.
        """
        if self.images:
            self.image = self.images.pop()
        else:
            self.active = False

class Level():
    """
    Класс Level представляет уровень игры, включая препятствия и плитки.

    Атрибуты:
        TILE_EMPTY, TILE_BRICK, TILE_STEEL, TILE_WATER, TILE_GRASS, TILE_FROZE: Константы типов плиток.
        TILE_SIZE (int): Размер плитки в пикселях.
        max_active_enemies (int): Максимальное количество врагов одновременно на уровне.
        obstacle_rects (list): Список прямоугольников препятствий на уровне.
    """
    # Константы типов плиток
    (TILE_EMPTY, TILE_BRICK, TILE_STEEL, TILE_WATER, TILE_GRASS, TILE_FROZE) = range(6)

    # Размер плитки в пикселях
    TILE_SIZE = 16

    def __init__(self, level_nr=None):
        """
        Инициализация уровня игры.

        Параметры:
            level_nr (int, optional): Номер уровня. Если больше 35, начинается с первого уровня.
        """
        global sprites

        self.max_active_enemies = 4  # Максимальное количество врагов на уровне

        tile_images = [
            pygame.Surface((8 * 2, 8 * 2)),
            sprites.subsurface(48 * 2, 64 * 2, 8 * 2, 8 * 2),
            sprites.subsurface(48 * 2, 72 * 2, 8 * 2, 8 * 2),
            sprites.subsurface(56 * 2, 72 * 2, 8 * 2, 8 * 2),
            sprites.subsurface(64 * 2, 64 * 2, 8 * 2, 8 * 2),
            sprites.subsurface(64 * 2, 64 * 2, 8 * 2, 8 * 2),
            sprites.subsurface(72 * 2, 64 * 2, 8 * 2, 8 * 2),
            sprites.subsurface(64 * 2, 72 * 2, 8 * 2, 8 * 2)
        ]

        self.tile_empty = tile_images[0]
        self.tile_brick = tile_images[1]
        self.tile_steel = tile_images[2]
        self.tile_grass = tile_images[3]
        self.tile_water = tile_images[4]
        self.tile_water1 = tile_images[4]
        self.tile_water2 = tile_images[5]
        self.tile_froze = tile_images[6]

        self.obstacle_rects = []
        self.loadLevel(level_nr)
        self.updateObstacleRects()

        gtimer.add(400, lambda: self.toggleWaves())

    def hitTile(self, pos, power=1, sound=False):
        """
        Обрабатывает попадание в плитку.

        Параметры:
            pos (tuple): Координаты плитки (x, y) в пикселях.
            power (int): Мощность попадания (1 или 2).
            sound (bool): Воспроизводить звук при попадании.

        Возвращает:
            bool: True, если снаряд остановился, иначе False.
        """

        global play_sounds, sounds

        for tile in self.mapr:
            if tile.topleft == pos:
                if tile.type == self.TILE_BRICK:
                    if play_sounds and sound:
                        sounds["brick"].play()
                    self.mapr.remove(tile)
                    self.updateObstacleRects()
                    return True
                elif tile.type == self.TILE_STEEL:
                    if play_sounds and sound:
                        sounds["steel"].play()
                    if power == 2:
                        self.mapr.remove(tile)
                        self.updateObstacleRects()
                    return True
                else:
                    return False

    def toggleWaves(self):
        """
        Переключает изображение воды между двумя состояниями для создания эффекта анимации.
        """
        if self.tile_water == self.tile_water1:
            self.tile_water = self.tile_water2
        else:
            self.tile_water = self.tile_water1

    def loadLevel(self, level_nr=1):
        """
        Загружает указанный уровень из файла.

        Параметры:
            level_nr (int): Номер уровня для загрузки.

        Возвращает:
            bool: True, если уровень успешно загружен, иначе False.
        """
        filename = f"levels/gameplay/{level_nr}"
        if not os.path.isfile(filename):
            return False

        level = []
        with open(filename, "r") as f:
            data = f.read().split("\n")

        self.mapr = []
        x, y = 0, 0
        for row in data:
            for ch in row:
                if ch == "#":
                    self.mapr.append(myRect(x, y, self.TILE_SIZE, self.TILE_SIZE, self.TILE_BRICK))
                elif ch == "@":
                    self.mapr.append(myRect(x, y, self.TILE_SIZE, self.TILE_SIZE, self.TILE_STEEL))
                elif ch == "~":
                    self.mapr.append(myRect(x, y, self.TILE_SIZE, self.TILE_SIZE, self.TILE_WATER))
                elif ch == "%":
                    self.mapr.append(myRect(x, y, self.TILE_SIZE, self.TILE_SIZE, self.TILE_GRASS))
                elif ch == "-":
                    self.mapr.append(myRect(x, y, self.TILE_SIZE, self.TILE_SIZE, self.TILE_FROZE))
                x += self.TILE_SIZE
            x = 0
            y += self.TILE_SIZE
        return True

    def draw(self, tiles=None):
        """
        Отрисовывает указанный уровень на поверхности.

        Параметры:
            tiles (list, optional): Список типов плиток для отображения. Если None, отображаются все плитки.
        """
        global screen

        # Рисуем линии запретной зоны
        pygame.draw.line(screen, (255, 0, 0), (64, 416), (64, 208), 3)
        pygame.draw.line(screen, (255, 0, 0), (64, 208), (352, 208), 3)
        pygame.draw.line(screen, (255, 0, 0), (352, 208), (352, 416), 3)

        if tiles is None:
            tiles = [self.TILE_BRICK, self.TILE_STEEL, self.TILE_WATER, self.TILE_GRASS, self.TILE_FROZE]

        for tile in self.mapr:
            if tile.type in tiles:
                if tile.type == self.TILE_BRICK:
                    screen.blit(self.tile_brick, tile.topleft)
                elif tile.type == self.TILE_STEEL:
                    screen.blit(self.tile_steel, tile.topleft)
                elif tile.type == self.TILE_WATER:
                    screen.blit(self.tile_water, tile.topleft)
                elif tile.type == self.TILE_FROZE:
                    screen.blit(self.tile_froze, tile.topleft)
                elif tile.type == self.TILE_GRASS:
                    screen.blit(self.tile_grass, tile.topleft)

    def updateObstacleRects(self):
        """
        Обновляет список прямоугольников препятствий, которые могут быть разрушены пулями.
        """
        global castle

        self.obstacle_rects = [castle.rect]

        for tile in self.mapr:
            if tile.type in (self.TILE_BRICK, self.TILE_STEEL, self.TILE_WATER):
                self.obstacle_rects.append(tile)

    def buildFortress(self, tile):
        """
        Строит стены вокруг замка из указанного типа плитки.

        Параметры:
            tile (int): Тип плитки для строительства стен.
        """
        positions = [
            (11 * self.TILE_SIZE, 23 * self.TILE_SIZE),
            (11 * self.TILE_SIZE, 24 * self.TILE_SIZE),
            (11 * self.TILE_SIZE, 25 * self.TILE_SIZE),
            (14 * self.TILE_SIZE, 23 * self.TILE_SIZE),
            (14 * self.TILE_SIZE, 24 * self.TILE_SIZE),
            (14 * self.TILE_SIZE, 25 * self.TILE_SIZE),
            (12 * self.TILE_SIZE, 23 * self.TILE_SIZE),
            (13 * self.TILE_SIZE, 23 * self.TILE_SIZE)
        ]

        obsolete = [rect for rect in self.mapr if rect.topleft in positions]

        for rect in obsolete:
            self.mapr.remove(rect)

        for pos in positions:
            self.mapr.append(myRect(pos[0], pos[1], self.TILE_SIZE, self.TILE_SIZE, tile))

        self.updateObstacleRects()

class Tank():

    # possible directions
    (DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT) = range(4)

    # states
    (STATE_SPAWNING, STATE_DEAD, STATE_ALIVE, STATE_EXPLODING) = range(4)

    # sides
    (SIDE_PLAYER, SIDE_ENEMY) = range(2)

    def __init__(self, level, side, position=None, direction=None, filename=None):
        """
        Инициализирует объект танка.

        Параметры:
            level (Level): Уровень, на котором находится танк.
            side (int): Сторона танка (игрок или враг).
            position (tuple, optional): Начальная позиция танка.
            direction (int, optional): Направление движения танка.
            filename (str, optional): Имя файла для дополнительной настройки танка.
        """
        global sprites

        self.health = 100  # Здоровье танка. 0 означает смерть.
        self.paralised = False  # Танк не может двигаться, но может стрелять и поворачиваться.
        self.paused = False  # Танк полностью обездвижен.
        self.shielded = False  # Танк защищен от пуль.

        self.speed = 5 * 2  # Скорость передвижения танка (пикселей за шаг).
        self.max_active_bullets = 1  # Максимальное количество активных пуль, которые танк может выпустить одновременно.

        self.side = side  # Сторона танка: игрок или враг.
        self.flash = 0  # Состояние мигания (0 - выкл., 1 - вкл.).

        self.superpowers = 0  # Уровень суперспособностей танка (0 - нет, 1-3 - разные уровни).
        self.bonus = None  # Один активный бонус для танка.

        self.controls = [pygame.K_SPACE, pygame.K_UP, pygame.K_RIGHT, pygame.K_DOWN,
                         pygame.K_LEFT]  # Клавиши управления.
        self.pressed = [False] * 4  # Нажатые клавиши направления.

        self.shield_images = [
            sprites.subsurface(0, 48 * 2, 16 * 2, 16 * 2),
            sprites.subsurface(16 * 2, 48 * 2, 16 * 2, 16 * 2)
        ]
        self.shield_image = self.shield_images[0]
        self.shield_index = 0

        self.spawn_images = [
            sprites.subsurface(32 * 2, 48 * 2, 16 * 2, 16 * 2),
            sprites.subsurface(48 * 2, 48 * 2, 16 * 2, 16 * 2)
        ]
        self.spawn_image = self.spawn_images[0]
        self.spawn_index = 0

        self.level = level

        if position is not None:
            self.rect = pygame.Rect(position, (26, 26))
        else:
            self.rect = pygame.Rect(0, 0, 26, 26)

        self.direction = direction if direction is not None else random.choice(
            [self.DIR_RIGHT, self.DIR_DOWN, self.DIR_LEFT])
        self.state = self.STATE_SPAWNING

        # Анимация появления танка
        self.timer_uuid_spawn = gtimer.add(100, lambda: self.toggleSpawnImage())
        self.timer_uuid_spawn_end = gtimer.add(1000, lambda: self.endSpawning())

    def endSpawning(self):
        """
        Завершает процесс появления танка.
        """
        self.state = self.STATE_ALIVE
        gtimer.destroy(self.timer_uuid_spawn_end)

    def toggleSpawnImage(self):
        """
        Переключает изображение появления танка на следующее.
        """
        if self.state != self.STATE_SPAWNING:
            gtimer.destroy(self.timer_uuid_spawn)
            return
        self.spawn_index = (self.spawn_index + 1) % len(self.spawn_images)
        self.spawn_image = self.spawn_images[self.spawn_index]

    def toggleShieldImage(self):
        """
        Переключает изображение щита на следующее.
        """
        if self.state != self.STATE_ALIVE:
            gtimer.destroy(self.timer_uuid_shield)
            return
        if self.shielded:
            self.shield_index = (self.shield_index + 1) % len(self.shield_images)
            self.shield_image = self.shield_images[self.shield_index]


    def draw(self):
        """
        Отображает танк на экране в зависимости от его состояния.
        """
        global screen
        if self.state == self.STATE_ALIVE:
            screen.blit(self.image, self.rect.topleft)
            if self.shielded:
                screen.blit(self.shield_image, [self.rect.left - 3, self.rect.top - 3])
        elif self.state == self.STATE_EXPLODING:
            self.explosion.draw()
        elif self.state == self.STATE_SPAWNING:
            screen.blit(self.spawn_image, self.rect.topleft)


    def explode(self):
        """
        Начинает процесс взрыва танка.
        """
        if self.state != self.STATE_DEAD:
            self.state = self.STATE_EXPLODING
            self.explosion = Explosion(self.rect.topleft)


    def fire(self, forced=False):
        """
        Стреляет пулей, если условия позволяют.

        Параметры:
            forced (bool): Если True, игнорирует ограничения на количество пуль.

        Возвращает:
            bool: True, если пуля была выпущена, иначе False.
        """
        global bullets, labels

        if self.state != self.STATE_ALIVE:
            gtimer.destroy(self.timer_uuid_fire)
            return False

        if self.paused:
            return False

        if not forced:
            active_bullets = sum(
                1 for bullet in bullets if bullet.owner_class == self and bullet.state == bullet.STATE_ACTIVE)
            if active_bullets >= self.max_active_bullets:
                return False

        bullet = Bullet(self.level, self.rect.topleft, self.direction)

        if self.superpowers > 0:
            bullet.speed = 5 * 8

        if self.superpowers > 2:
            bullet.power = 2

        bullet.owner = self.side
        bullet.owner_class = self
        bullets.append(bullet)
        return True


    def rotate(self, direction, fix_position=True):
        """
        Поворачивает танк и корректирует его позицию.

        Параметры:
            direction (int): Направление, в котором нужно повернуть танк.
            fix_position (bool): Флаг для коррекции позиции танка.
        """
        self.direction = direction

        if direction == self.DIR_UP:
            self.image = self.image_up
        elif direction == self.DIR_RIGHT:
            self.image = self.image_right
        elif direction == self.DIR_DOWN:
            self.image = self.image_down
        elif direction == self.DIR_LEFT:
            self.image = self.image_left

        if fix_position:
            new_x = self.nearest(self.rect.left, 8) + 3
            new_y = self.nearest(self.rect.top, 8) + 3

            if abs(self.rect.left - new_x) < 5:
                self.rect.left = new_x

            if abs(self.rect.top - new_y) < 5:
                self.rect.top = new_y


    def turnAround(self):
        """
        Разворачивает танк в противоположное направление.
        """
        if self.direction in (self.DIR_UP, self.DIR_RIGHT):
            self.rotate(self.direction + 2, False)
        else:
            self.rotate(self.direction - 2, False)


    def update(self, time_passed):
        """
        Обновляет таймер и проверяет состояние взрыва танка.

        Параметры:
            time_passed (int): Прошедшее время с последнего обновления.
        """
        if self.state == self.STATE_EXPLODING:
            if not self.explosion.active:
                self.state = self.STATE_DEAD
                del self.explosion


    def nearest(self, num, base):
        """
        Округляет число до ближайшего кратного базовому значению.

        Параметры:
            num (int): Число для округления.
            base (int): Базовое значение для кратности.

        Возвращает:
            int: Округленное число.
        """
        return int(round(num / base) * base)


    def bulletImpact(self, friendly_fire=False, damage=100, tank=None):
        """
        Обрабатывает попадание пули в танк.

        Параметры:
            friendly_fire (bool): Флаг дружественного огня.
            damage (int): Урон от пули.
            tank (Tank): Объект танка, выпустившего пулю.

        Возвращает:
            bool: True, если пуля должна быть уничтожена при попадании, иначе False.
        """
        global play_sounds, sounds

        if self.shielded:
            return True

        if not friendly_fire:
            self.health -= damage
            if self.health < 1:
                if self.side == self.SIDE_ENEMY:
                    tank.trophies[f"enemy{self.type}"] += 1
                    points = (self.type + 1) * 100
                    tank.score += points
                    if play_sounds:
                        sounds["explosion"].play()
                    labels.append(Label(self.rect.topleft, str(points), 500))

                self.explode()
            return True

        if self.side == self.SIDE_ENEMY:
            return False
        elif self.side == self.SIDE_PLAYER:
            return False


    def setParalised(self, paralised=True):
        """
        Устанавливает состояние парализации танка.

        Параметры:
            paralised (bool): Флаг состояния парализации.
        """
        if self.state != self.STATE_ALIVE:
            gtimer.destroy(self.timer_uuid_paralise)
            return
        self.paralised = paralised


class Enemy(Tank):

    (TYPE_BASIC, TYPE_FAST, TYPE_POWER, TYPE_ARMOR) = range(4)

    def __init__(self, level, type, position=None, direction=None, filename=None):
        """
        Конструктор класса Enemy, инициализирующий параметры вражеского танка.

        Параметры:
            level (Level): Уровень, на котором находится враг.
            type (int): Тип вражеского танка.
            position (tuple): Позиция танка (опционально).
            direction (int): Направление танка (опционально).
            filename (str): Имя файла с ресурсами для танка (опционально).
        """
        Tank.__init__(self, level, type, position=None, direction=None, filename=None)

        global enemies, sprites

        # Если True, танк не стреляет
        self.bullet_queued = False

        # Случайный выбор типа танка
        if len(level.enemies_left) > 0:
            self.type = level.enemies_left.pop()
        else:
            self.state = self.STATE_DEAD
            return

        if self.type == self.TYPE_BASIC:
            self.speed = 5 * 1
        elif self.type == self.TYPE_FAST:
            self.speed = 5 * 3
        elif self.type == self.TYPE_POWER:
            self.superpowers = 5 * 1
        elif self.type == self.TYPE_ARMOR:
            self.health = 400

        # 1 из 5 шансов, что танк будет с бонусом, если нет других бонусных танков
        if random.randint(1, 5) == 1:
            self.bonus = True
            for enemy in enemies:
                if enemy.bonus:
                    self.bonus = False
                    break

        images = [
            sprites.subsurface(32 * 2, 0, 13 * 2, 15 * 2),
            sprites.subsurface(48 * 2, 0, 13 * 2, 15 * 2),
            sprites.subsurface(64 * 2, 0, 13 * 2, 15 * 2),
            sprites.subsurface(80 * 2, 0, 13 * 2, 15 * 2),
            sprites.subsurface(32 * 2, 16 * 2, 13 * 2, 15 * 2),
            sprites.subsurface(48 * 2, 16 * 2, 13 * 2, 15 * 2),
            sprites.subsurface(64 * 2, 16 * 2, 13 * 2, 15 * 2),
            sprites.subsurface(80 * 2, 16 * 2, 13 * 2, 15 * 2)
        ]

        self.image = images[self.type + 0]

        self.image_up = self.image
        self.image_left = pygame.transform.rotate(self.image, 90)
        self.image_down = pygame.transform.rotate(self.image, 180)
        self.image_right = pygame.transform.rotate(self.image, 270)

        if self.bonus:
            self.image1_up = self.image_up
            self.image1_left = self.image_left
            self.image1_down = self.image_down
            self.image1_right = self.image_right

            self.image2 = images[self.type + 4]
            self.image2_up = self.image2
            self.image2_left = pygame.transform.rotate(self.image2, 90)
            self.image2_down = pygame.transform.rotate(self.image2, 180)
            self.image2_right = pygame.transform.rotate(self.image2, 270)

        self.rotate(self.direction, False)

        if position == None:
            self.rect.topleft = self.getFreeSpawningPosition()
            if not self.rect.topleft:
                self.state = self.STATE_DEAD
                return

        # Список координат карты, куда танк должен двигаться
        self.path = self.generatePath(self.direction)

        # Продолжительность между выстрелами
        self.timer_uuid_fire = gtimer.add(1000, lambda: self.fire())

        # Включение мигания для бонусных танков
        if self.bonus:
            self.timer_uuid_flash = gtimer.add(200, lambda: self.toggleFlash())

    def toggleFlash(self):
        """
        Переключает состояние мигания для бонусных танков.
        """
        if self.state not in (self.STATE_ALIVE, self.STATE_SPAWNING):
            gtimer.destroy(self.timer_uuid_flash)
            return
        self.flash = not self.flash
        if self.flash:
            self.image_up = self.image2_up
            self.image_right = self.image2_right
            self.image_down = self.image2_down
            self.image_left = self.image2_left
        else:
            self.image_up = self.image1_up
            self.image_right = self.image1_right
            self.image_down = self.image1_down
            self.image_left = self.image1_left
        self.rotate(self.direction, False)

    def getFreeSpawningPosition(self):
        """
        Получить доступную позицию для появления врага.

        Возвращает:
            tuple: Координаты позиции для появления врага или False, если позиции нет.
        """
        global players, enemies

        available_positions = [
            [(self.level.TILE_SIZE * 2 - self.rect.width) / 2, (self.level.TILE_SIZE * 2 - self.rect.height) / 2],
            [12 * self.level.TILE_SIZE + (self.level.TILE_SIZE * 2 - self.rect.width) / 2,
             (self.level.TILE_SIZE * 2 - self.rect.height) / 2],
            [24 * self.level.TILE_SIZE + (self.level.TILE_SIZE * 2 - self.rect.width) / 2,
             (self.level.TILE_SIZE * 2 - self.rect.height) / 2],
        ]

        random.shuffle(available_positions)

        for pos in available_positions:

            enemy_rect = pygame.Rect(pos, [26, 26])

            # Проверка на столкновения с другими врагами
            collision = False
            for enemy in enemies:
                if enemy_rect.colliderect(enemy.rect):
                    collision = True
                    continue

            if collision:
                continue

            # Проверка на столкновения с игроками
            collision = False
            for player in players:
                if enemy_rect.colliderect(player.rect):
                    collision = True
                    continue

            if collision:
                continue

            return pos
        return False

    def move(self):
        """
        Перемещает врага, если это возможно.
        """
        global players, enemies, bonuses

        if self.state != self.STATE_ALIVE or self.paused or self.paralised:
            return

        if self.path == []:
            self.path = self.generatePath(None, True)

        new_position = self.path.pop(0)

        # Движение врага
        if self.direction == self.DIR_UP:
            if new_position[1] < 0:
                self.path = self.generatePath(self.direction, True)
                return
        elif self.direction == self.DIR_RIGHT:
            if new_position[0] > (416 - 26):
                self.path = self.generatePath(self.direction, True)
                return
        elif self.direction == self.DIR_DOWN:
            if new_position[1] > (416 - 26):
                self.path = self.generatePath(self.direction, True)
                return
        elif self.direction == self.DIR_LEFT:
            if new_position[0] < 0:
                self.path = self.generatePath(self.direction, True)
                return

        new_rect = pygame.Rect(new_position, [26, 26])

        # Проверка на столкновения с препятствиями
        if new_rect.collidelist(self.level.obstacle_rects) != -1:
            self.path = self.generatePath(self.direction, True)
            return

        # Проверка на столкновения с другими врагами
        for enemy in enemies:
            if enemy != self and new_rect.colliderect(enemy.rect):
                self.turnAround()
                self.path = self.generatePath(self.direction)
                return

        # Проверка на столкновения с игроками
        for player in players:
            if new_rect.colliderect(player.rect):
                self.turnAround()
                self.path = self.generatePath(self.direction)
                return

        # Проверка на столкновения с бонусами
        for bonus in bonuses:
            if new_rect.colliderect(bonus.rect):
                bonuses.remove(bonus)

        # Если столкновений нет, переместить врага
        self.rect.topleft = new_rect.topleft

    def update(self, time_passed):
        """
        Обновление состояния врага.
        """
        Tank.update(self, time_passed)
        if self.state == self.STATE_ALIVE and not self.paused:
            self.move()

    def generatePath(self, direction=None, fix_direction=False):
        """ Генерация пути для движения врага в зависимости от направления и структуры уровня.

        Если направление указано, пытаемся продолжить движение в этом направлении,
        иначе выбирается случайное направление.

        Параметры:
            direction (int): Предпочтительное направление движения (опционально).
            fix_direction (bool): Флаг для выравнивания позиции танка по сетке после определения пути (опционально).

        Возвращает:
            list: Список позиций, представляющих путь.
        """

        all_directions = [self.DIR_UP, self.DIR_RIGHT, self.DIR_DOWN, self.DIR_LEFT]

        if direction is None:
            if self.direction in [self.DIR_UP, self.DIR_RIGHT]:
                opposite_direction = self.direction + 2
            else:
                opposite_direction = self.direction - 2
            directions = all_directions
            random.shuffle(directions)
            directions.remove(opposite_direction)
            directions.append(opposite_direction)
        else:
            if direction in [self.DIR_UP, self.DIR_RIGHT]:
                opposite_direction = direction + 2
            else:
                opposite_direction = direction - 2

            directions = all_directions
            random.shuffle(directions)
            directions.remove(opposite_direction)
            directions.remove(direction)
            directions.insert(0, direction)
            directions.append(opposite_direction)

        # Работаем с единицами сетки (шагами), а не пикселями
        x = int(round(self.rect.left / 16))
        y = int(round(self.rect.top / 16))

        new_direction = None

        for direction in directions:
            if direction == self.DIR_UP and y > 1:
                new_pos_rect = self.rect.move(0, -8)
                if new_pos_rect.collidelist(self.level.obstacle_rects) == -1:
                    new_direction = direction
                    break
            elif direction == self.DIR_RIGHT and x < 24:
                new_pos_rect = self.rect.move(8, 0)
                if new_pos_rect.collidelist(self.level.obstacle_rects) == -1:
                    new_direction = direction
                    break
            elif direction == self.DIR_DOWN and y < 24:
                new_pos_rect = self.rect.move(0, 8)
                if new_pos_rect.collidelist(self.level.obstacle_rects) == -1:
                    new_direction = direction
                    break
            elif direction == self.DIR_LEFT and x > 1:
                new_pos_rect = self.rect.move(-8, 0)
                if new_pos_rect.collidelist(self.level.obstacle_rects) == -1:
                    new_direction = direction
                    break

        # Если нет доступных направлений, разворачиваемся
        if new_direction is None:
            new_direction = opposite_direction

        # Выравниваем позицию танка
        if fix_direction and new_direction == self.direction:
            fix_direction = False

        self.rotate(new_direction, fix_direction)

        positions = []

        x = self.rect.left
        y = self.rect.top

        if new_direction in (self.DIR_RIGHT, self.DIR_LEFT):
            axis_fix = self.nearest(y, 16) - y
        else:
            axis_fix = self.nearest(x, 16) - x
        axis_fix = 0

        pixels = self.nearest(random.randint(1, 12) * 32, 32) + axis_fix + 3

        if new_direction == self.DIR_UP:
            for px in range(0, pixels, self.speed):
                positions.append([x, y - px])
        elif new_direction == self.DIR_RIGHT:
            for px in range(0, pixels, self.speed):
                positions.append([x + px, y])
        elif new_direction == self.DIR_DOWN:
            for px in range(0, pixels, self.speed):
                positions.append([x, y + px])
        elif new_direction == self.DIR_LEFT:
            for px in range(0, pixels, self.speed):
                positions.append([x - px, y])

        return positions

class Player(Tank):
    """
    Класс Player — реализация игрока.
    """

    def __init__(self, level, type, position=None, direction=None, filename=None):
        """
        Конструктор игрока.

        Параметры:
            level: объект уровня, в котором находится игрок.
            type: тип игрока.
            position: начальная позиция игрока.
            direction: начальное направление игрока.
            filename: имя файла спрайта игрока.
        """
        Tank.__init__(self, level, type, position=None, direction=None, filename=None)

        global sprites

        if filename is None:
            filename = (0, 0, 16 * 2, 16 * 2)

        self.start_position = position
        self.start_direction = direction

        self.lives = 3  # Количество жизней игрока.

        # Счёт игрока.
        self.score = 0

        # Счётчики трофеев игрока на уровне.
        self.trophies = {
            "bonus": 0,
            "enemy0": 0,
            "enemy1": 0,
            "enemy2": 0,
            "enemy3": 0
        }

        self.image = sprites.subsurface(filename)
        self.image_up = self.image
        self.image_left = pygame.transform.rotate(self.image, 90)
        self.image_down = pygame.transform.rotate(self.image, 180)
        self.image_right = pygame.transform.rotate(self.image, 270)

        if direction is None:
            self.rotate(self.DIR_UP, False)
        else:
            self.rotate(direction, False)

    def move(self, direction):
        """
        Перемещение игрока.

        Параметры:
            direction: направление движения.
        """
        global obs_flag_player_collision
        global players, enemies, bonuses

        if self.state == self.STATE_EXPLODING:
            if not self.explosion.active:
                self.state = self.STATE_DEAD
                del self.explosion

        if self.state != self.STATE_ALIVE:
            return

        # Поворот игрока.
        if self.direction != direction:
            self.rotate(direction)

        if self.paralised:
            return

        # Перемещение игрока (с ограничением зоны вокруг базы).
        if direction == self.DIR_UP:
            new_position = [self.rect.left, self.rect.top - self.speed]
            if new_position[1] < 0 + 416 // 2:
                obs_flag_player_collision = 1
                return
        elif direction == self.DIR_RIGHT:
            new_position = [self.rect.left + self.speed, self.rect.top]
            if new_position[0] > (416 - 26) - 32 * 2:
                obs_flag_player_collision = 1
                return
        elif direction == self.DIR_DOWN:
            new_position = [self.rect.left, self.rect.top + self.speed]
            if new_position[1] > (416 - 26):
                obs_flag_player_collision = 1
                return
        elif direction == self.DIR_LEFT:
            new_position = [self.rect.left - self.speed, self.rect.top]
            if new_position[0] < 0 + 32 * 2:
                obs_flag_player_collision = 1
                return

        player_rect = pygame.Rect(new_position, [26, 26])

        # Проверка на столкновение с препятствиями.
        if player_rect.collidelist(self.level.obstacle_rects) != -1:
            obs_flag_player_collision = 1
            return

        # Проверка на столкновение с другими игроками.
        for player in players:
            if player != self and player.state == player.STATE_ALIVE and player_rect.colliderect(player.rect):
                obs_flag_player_collision = 1
                return

        # Проверка на столкновение с врагами.
        for enemy in enemies:
            if player_rect.colliderect(enemy.rect):
                obs_flag_player_collision = 1
                return

        # Проверка на сбор бонусов.
        for bonus in bonuses:
            if player_rect.colliderect(bonus.rect):
                self.bonus = bonus

        # Перемещение игрока, если нет столкновений.
        self.rect.topleft = (new_position[0], new_position[1])

    def reset(self, pos):
        """
        Сброс параметров игрока.

        Параметры:
            pos: новая стартовая позиция игрока.
        """
        self.start_position = pos
        self.start_direction = random.randint(0, 3)
        self.rotate(self.start_direction, False)
        self.rect.topleft = self.start_position
        self.superpowers = 0
        self.max_active_bullets = 1
        self.health = 100
        self.paralised = False
        self.paused = False
        self.pressed = [False] * 4
        self.state = self.STATE_ALIVE


class Game():
    """
       Класс Game отвечает за управление основной логикой игры, включая загрузку ресурсов, обработку бонусов, защиту игроков и управление игровым процессом.
    """

    # Константы направлений
    (DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT) = range(4)

    TILE_SIZE = 16  # Размер одной плитки на карте

    def __init__(self):
        """
            Конструктор класса Game. Инициализирует основные параметры игры, загружает ресурсы и устанавливает начальные значения.
        """
        global screen, sprites, play_sounds, sounds

        # Центрируем окно игры
        os.environ['SDL_VIDEO_WINDOW_POS'] = 'center'

        if play_sounds:
            pygame.mixer.pre_init(44100, -16, 1, 512)

        pygame.init()


        pygame.display.set_caption("Battle City")

        size = width, height = 480, 416

        if "-f" in sys.argv[1:]:
            screen = pygame.display.set_mode(size, pygame.FULLSCREEN)
        else:
            screen = pygame.display.set_mode(size)

        self.clock = pygame.time.Clock()

        sprites = pygame.transform.scale(pygame.image.load("images/sprites.gif"), [192, 224])

        pygame.display.set_icon(sprites.subsurface(0, 0, 13*2, 13*2))

        # Загрузка звуков
        if play_sounds:
            pygame.mixer.init(44100, -16, 1, 512)

            sounds["start"] = pygame.mixer.Sound("sounds/gamestart.ogg")
            sounds["end"] = pygame.mixer.Sound("sounds/gameover.ogg")
            sounds["score"] = pygame.mixer.Sound("sounds/score.ogg")
            sounds["bg"] = pygame.mixer.Sound("sounds/background.ogg")
            sounds["fire"] = pygame.mixer.Sound("sounds/fire.ogg")
            sounds["bonus"] = pygame.mixer.Sound("sounds/bonus.ogg")
            sounds["explosion"] = pygame.mixer.Sound("sounds/explosion.ogg")
            sounds["brick"] = pygame.mixer.Sound("sounds/brick.ogg")
            sounds["steel"] = pygame.mixer.Sound("sounds/steel.ogg")

        # Иконки для жизней и флага
        self.enemy_life_image = sprites.subsurface(81*2, 57*2, 7*2, 7*2)
        self.player_life_image = sprites.subsurface(89*2, 56*2, 7*2, 8*2)
        self.flag_image = sprites.subsurface(64*2, 49*2, 16*2, 15*2)

        # Изображение игрока для начального экрана
        self.player_image = pygame.transform.rotate(sprites.subsurface(0, 0, 13*2, 13*2), 270)

        # Переменная для временной остановки врагов
        self.timefreeze = False

        # Загрузка шрифта
        self.font = pygame.font.Font("fonts/prstart.ttf", 16)

        # Предварительная отрисовка текста "GAME OVER"
        self.im_game_over = pygame.Surface((64, 40))
        self.im_game_over.set_colorkey((0, 0, 0))
        self.im_game_over.blit(self.font.render("GAME", False, (127, 64, 64)), [0, 0])
        self.im_game_over.blit(self.font.render("OVER", False, (127, 64, 64)), [0, 20])
        self.game_over_y = 416+40

        # Количество игроков
        self.nr_of_players = 1
        self.available_positions = []

        # Очистка глобальных списков
        del players[:]
        del bullets[:]
        del enemies[:]
        del bonuses[:]


    def triggerBonus(self, bonus, player):
        """
        Активировать бонус и применить его к игроку.

        Параметры:
            bonus: объект бонуса.
            player: игрок, получивший бонус.
        """

        global enemies, labels, play_sounds, sounds

        if play_sounds:
            sounds["bonus"].play()

        player.trophies["bonus"] += 1
        player.score += 500

        if bonus.bonus == bonus.BONUS_GRENADE:
            for enemy in enemies:
                enemy.explode()
        elif bonus.bonus == bonus.BONUS_HELMET:
            self.shieldPlayer(player, True, 10000)
        elif bonus.bonus == bonus.BONUS_SHOVEL:
            self.level.buildFortress(self.level.TILE_STEEL)
            gtimer.add(10000, lambda: self.level.buildFortress(self.level.TILE_BRICK), 1)
        elif bonus.bonus == bonus.BONUS_STAR:
            player.superpowers += 1
            if player.superpowers == 2:
                player.max_active_bullets = 2
        elif bonus.bonus == bonus.BONUS_TANK:
            player.lives += 1
        elif bonus.bonus == bonus.BONUS_TIMER:
            self.toggleEnemyFreeze(True)
            gtimer.add(10000, lambda: self.toggleEnemyFreeze(False), 1)
        bonuses.remove(bonus)

        labels.append(Label(bonus.rect.topleft, "500", 500))

    def shieldPlayer(self, player, shield=True, duration=None):
        """
        Добавить или убрать защиту игрока.

        Параметры:
            player: объект игрока.
            shield: True для добавления защиты, False для её снятия.
            duration: продолжительность защиты в миллисекундах.
        """
        player.shielded = shield
        if shield:
            player.timer_uuid_shield = gtimer.add(100, lambda: player.toggleShieldImage())
        else:
            gtimer.destroy(player.timer_uuid_shield)

        if shield and duration != None:
            gtimer.add(duration, lambda: self.shieldPlayer(player, False), 1)


    def spawnEnemy(self):
        """
        Появление нового врага на карте.

        Враг появляется только в следующих случаях:
        - Если есть враги в очереди.
        - Если количество активных врагов не превышает установленный максимум.
        - Если не активирован эффект временной остановки (timefreeze).
        """

        global enemies

        if len(enemies) >= self.level.max_active_enemies:
            return
        if len(self.level.enemies_left) < 1 or self.timefreeze:
            return
        enemy = Enemy(self.level, 1)

        enemies.append(enemy)


    def respawnPlayer(self, player, clear_scores = False):
        """
        Перезапуск игрока (возвращение в игру).

        Параметры:
            player: объект игрока.
            clear_scores (bool): если True, сбрасывает все очки и трофеи игрока.
        """
        n = random.randint(0, len(self.available_positions) - 1)
        [kx, ky] = self.available_positions[n]
        x = kx * self.TILE_SIZE + (self.TILE_SIZE * 2 - 26) / 2
        y = ky * self.TILE_SIZE + (self.TILE_SIZE * 2 - 26) / 2
        pos = [x, y]

        player.reset(pos)

        if clear_scores:
            player.trophies = {
                "bonus": 0, "enemy0": 0, "enemy1": 0, "enemy2": 0, "enemy3": 0
            }

        self.shieldPlayer(player, True, 4000)

    def gameOver(self):
        """
        Завершение игры. Возвращение в меню или перезапуск уровня.
        """

        global play_sounds, sounds

        for player in players:
            player.lives = 3  # Установка начального количества жизней


        if play_sounds:
            for sound in sounds:
                sounds[sound].stop()
            sounds["end"].play()

        self.game_over_y = 416+40

        self.game_over = True
        #gtimer.add(3000, lambda :self.showScores(), 1)
        if self.game_over:
            self.stage = 0
            self.nextLevel()
            # self.gameOverScreen() #   DONE FLAG
        else:
            self.nextLevel()

    def gameOverScreen(self):
        """
        Экран завершения игры ("GAME OVER").
        """

        global screen

        # остановить основной цикл игры (если есть)
        self.running = False

        screen.fill([0, 0, 0])

        self.writeInBricks("game", [125, 140])
        self.writeInBricks("over", [125, 220])
        pygame.display.flip()

        while True:
            time_passed = self.clock.tick(50)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        self.showMenu()
                        return

    def showMenu(self):
        """
        Показ игрового меню. Перерисовка экрана выполняется при нажатии клавиш вверх или вниз.
        Нажатие Enter запускает игру.
        """

        global players, screen, gtimer

        # остановить основной цикл игры (если есть)
        self.running = False

        # очистить все таймеры
        del gtimer.timers[:]

        # установить текущий этап на 0
        self.stage = 0
        self.nr_of_players = 1
        del players[:]
        self.nextLevel()


    def reloadPlayers(self):
        """
        Инициализация игроков.

        Если игроки уже существуют, просто сбрасывает их состояние.
        """

        global players

        if len(players) == 0:

            x = 8 * self.TILE_SIZE + (self.TILE_SIZE * 2 - 26) / 2
            y = 24 * self.TILE_SIZE + (self.TILE_SIZE * 2 - 26) / 2

            player = Player(
                self.level, 0, [x, y], self.DIR_UP, (0, 0, 13*2, 13*2)
            )
            players.append(player)

        for player in players:
            player.level = self.level
            self.respawnPlayer(player, True)

    def showScores(self):
        """
        Показать результаты уровня.

        Отображает статистику игроков, количество уничтоженных танков и очки.
        Также обновляет рекорд, если текущие очки превышают его.
        """

        global screen, sprites, players, play_sounds, sounds

        # Остановка главного игрового цикла, если он был запущен
        self.running = False

        # Очистка всех таймеров
        del gtimer.timers[:]

        # Остановка всех звуков
        if play_sounds:
            for sound in sounds:
                sounds[sound].stop()

        # Загрузка рекорда
        hiscore = self.loadHiscore()

        # Обновление рекорда, если это необходимо
        if players[0].score > hiscore:
            hiscore = players[0].score
            self.saveHiscore(hiscore)
        if self.nr_of_players == 2 and players[1].score > hiscore:
            hiscore = players[1].score
            self.saveHiscore(hiscore)

        # Изображения танков и стрелок
        img_tanks = [
            sprites.subsurface(32*2, 0, 13*2, 15*2),
            sprites.subsurface(48*2, 0, 13*2, 15*2),
            sprites.subsurface(64*2, 0, 13*2, 15*2),
            sprites.subsurface(80*2, 0, 13*2, 15*2)
        ]

        img_arrows = [
            sprites.subsurface(81*2, 48*2, 7*2, 7*2),
            sprites.subsurface(88*2, 48*2, 7*2, 7*2)
        ]

        # Очистка экрана
        screen.fill([0, 0, 0])

        # Цвета
        black = pygame.Color("black")
        white = pygame.Color("white")
        purple = pygame.Color(127, 64, 64)
        pink = pygame.Color(191, 160, 128)

        # Отображение рекорда и текущего уровня
        screen.blit(self.font.render("HI-SCORE", False, purple), [105, 35])
        screen.blit(self.font.render(str(hiscore), False, pink), [295, 35])

        screen.blit(self.font.render("STAGE"+str(self.stage).rjust(3), False, white), [170, 65])

        screen.blit(self.font.render("I-PLAYER", False, purple), [25, 95])

        # Очки первого игрока
        screen.blit(self.font.render(str(players[0].score).rjust(8), False, pink), [25, 125])

        if self.nr_of_players == 2:
            screen.blit(self.font.render("II-PLAYER", False, purple), [310, 95])

            # Если два игрока, отображаем второго
            screen.blit(self.font.render(str(players[1].score).rjust(8), False, pink), [325, 125])

        # Отображение танков и стрелок
        for i in range(4):
            screen.blit(img_tanks[i], [226, 160+(i*45)])
            screen.blit(img_arrows[0], [206, 168+(i*45)])
            if self.nr_of_players == 2:
                screen.blit(img_arrows[1], [258, 168+(i*45)])

        screen.blit(self.font.render("TOTAL", False, white), [70, 335])

        # Общее количество очков
        pygame.draw.line(screen, white, [170, 330], [307, 330], 4)

        pygame.display.flip()

        self.clock.tick(2)

        interval = 5

        # Интервал отображения анимации
        for i in range(4):

            # Подсчёт очков и количества уничтоженных танков
            tanks = players[0].trophies["enemy"+str(i)]

            for n in range(tanks+1):
                if n > 0 and play_sounds:
                    sounds["score"].play()

                # стереть предыдущий текст
                screen.blit(self.font.render(str(n-1).rjust(2), False, black), [170, 168+(i*45)])
                # вывести новое количество врагов
                screen.blit(self.font.render(str(n).rjust(2), False, white), [170, 168+(i*45)])
                # стереть предыдущий текст
                screen.blit(self.font.render(str((n-1) * (i+1) * 100).rjust(4)+" PTS", False, black), [25, 168+(i*45)])
                # вывести новое общее количество очков за врага
                screen.blit(self.font.render(str(n * (i+1) * 100).rjust(4)+" PTS", False, white), [25, 168+(i*45)])
                pygame.display.flip()
                self.clock.tick(interval)

            if self.nr_of_players == 2:
                tanks = players[1].trophies["enemy"+str(i)]

                for n in range(tanks+1):

                    if n > 0 and play_sounds:
                        sounds["score"].play()

                    screen.blit(self.font.render(str(n-1).rjust(2), False, black), [277, 168+(i*45)])
                    screen.blit(self.font.render(str(n).rjust(2), False, white), [277, 168+(i*45)])

                    screen.blit(self.font.render(str((n-1) * (i+1) * 100).rjust(4)+" PTS", False, black), [325, 168+(i*45)])
                    screen.blit(self.font.render(str(n * (i+1) * 100).rjust(4)+" PTS", False, white), [325, 168+(i*45)])

                    pygame.display.flip()
                    self.clock.tick(interval)

            self.clock.tick(interval)

        # Общее количество танков
        tanks = sum([i for i in players[0].trophies.values()]) - players[0].trophies["bonus"]
        screen.blit(self.font.render(str(tanks).rjust(2), False, white), [170, 335])
        if self.nr_of_players == 2:
            tanks = sum([i for i in players[1].trophies.values()]) - players[1].trophies["bonus"]
            screen.blit(self.font.render(str(tanks).rjust(2), False, white), [277, 335])

        pygame.display.flip()

        # Задержка на 2 секунды
        self.clock.tick(1)
        self.clock.tick(1)

        # Переход на следующий уровень или экран завершения игры
        if self.game_over:
            self.gameOverScreen()
        else:
            self.nextLevel()


    def draw(self):
        """
            Отображение основных элементов игры на экране.
            Отрисовывает уровень, замок, игроков, врагов, пули, бонусы и боковую панель.
        """
        global screen, castle, players, enemies, bullets, bonuses

        # Очистка экрана
        screen.fill([0, 0, 0])

        # Отрисовка основных элементов уровня
        self.level.draw([self.level.TILE_EMPTY, self.level.TILE_BRICK, self.level.TILE_STEEL, self.level.TILE_FROZE, self.level.TILE_WATER])
        # Отрисовка замка
        castle.draw()
        # Отрисовка замка
        for enemy in enemies:
            enemy.draw()
        # Отрисовка игроков
        for player in players:
            player.draw()
        # Отрисовка пуль
        for bullet in bullets:
            bullet.draw()
        # Отрисовка бонусов
        for bonus in bonuses:
            bonus.draw()
        # Отрисовка элементов, перекрывающих основной уровень (например, трава)
        self.level.draw([self.level.TILE_GRASS])

        # Отображение текста "GAME OVER", если игра завершена
        if self.game_over:
            if self.game_over_y > 188:
                self.game_over_y -= 4
            screen.blit(self.im_game_over, [176, self.game_over_y])  # 176=(416-64)/2
        # Отрисовка боковой панели
        self.drawSidebar()
        # Обновление дисплея
        pygame.display.flip()

    def drawSidebar(self):
        """
            Отрисовка боковой панели с жизнями игроков, количеством оставшихся врагов и текущим уровнем.
        """
        global screen, players, enemies

        # Очистка боковой панели
        x = 416
        y = 0
        screen.fill([100, 100, 100], pygame.Rect([416, 0], [64, 416]))

        xpos = x + 16
        ypos = y + 16

        # Отображение оставшихся жизней врагов
        for n in range(len(self.level.enemies_left) + len(enemies)):
            screen.blit(self.enemy_life_image, [xpos, ypos])
            if n % 2 == 1:
                xpos = x + 16
                ypos+= 17
            else:
                xpos += 17

        # Отображение жизней игроков
        if pygame.font.get_init():
            text_color = pygame.Color('black')
            for n in range(len(players)):
                if n == 0:
                    screen.blit(self.font.render(str(n+1)+"P", False, text_color), [x+16, y+200])
                    screen.blit(self.font.render(str(players[n].lives), False, text_color), [x+31, y+215])
                    screen.blit(self.player_life_image, [x+17, y+215])
                else:
                    screen.blit(self.font.render(str(n+1)+"P", False, text_color), [x+16, y+240])
                    screen.blit(self.font.render(str(players[n].lives), False, text_color), [x+31, y+255])
                    screen.blit(self.player_life_image, [x+17, y+255])
            # Отображение уровня (флаг и номер уровня)
            screen.blit(self.flag_image, [x+17, y+280])
            screen.blit(self.font.render(str(self.stage), False, text_color), [x+17, y+312])


    def drawIntroScreen(self, put_on_surface = True):
        """
            Отображение стартового экрана (меню).
            @param put_on_surface: Если True, выполняется обновление дисплея.
        """

        global screen
        # Очистка экрана
        screen.fill([0, 0, 0])

        if pygame.font.get_init():
            hiscore = self.loadHiscore()
            # Отображение рекорда и пунктов меню
            screen.blit(self.font.render("HI- "+str(hiscore), True, pygame.Color('white')), [170, 35])

            screen.blit(self.font.render("1 PLAYER", True, pygame.Color('white')), [165, 250])
            screen.blit(self.font.render("2 PLAYERS", True, pygame.Color('white')), [165, 275])

            screen.blit(self.font.render("(c) 1980 1985 NAMCO LTD.", True, pygame.Color('white')), [50, 350])
            screen.blit(self.font.render("ALL RIGHTS RESERVED", True, pygame.Color('white')), [85, 380])

        # Отображение значка игрока
        if self.nr_of_players == 1:
            screen.blit(self.player_image, [125, 245])
        elif self.nr_of_players == 2:
            screen.blit(self.player_image, [125, 270])
        # Отображение названия игры
        self.writeInBricks("battle", [65, 80])
        self.writeInBricks("city", [129, 160])

        if put_on_surface:
            pygame.display.flip()

    def animateIntroScreen(self):
        """
            Анимация стартового экрана: слайд снизу вверх.
            Нажатие клавиши Enter пропускает анимацию.
        """

        global screen

        self.drawIntroScreen(False)
        screen_cp = screen.copy()

        screen.fill([0, 0, 0])

        y = 416
        while (y > 0):
            time_passed = self.clock.tick(50)
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        y = 0
                        break

            screen.blit(screen_cp, [0, y])
            pygame.display.flip()
            y -= 5

        screen.blit(screen_cp, [0, 0])
        pygame.display.flip()


    def chunks(self, l, n):
        """
            Разбивает строку текста на части заданного размера.
            :param l: входная строка
            :param n: размер (число символов) каждой части
            :return: список частей строки
        """
        return [l[i:i+n] for i in range(0, len(l), n)]

    def writeInBricks(self, text, pos):
        """
            Пишет заданный текст в формате "шрифтом из кирпичей".
            Доступны только буквы, которые образуют слова "Battle City" и "Game Over".
            Каждая буква состоит из 7x7 кирпичей.
            :param text: текст для отображения
            :param pos: позиция (x, y), где будет начата отрисовка текста
        """

        global screen, sprites

        bricks = sprites.subsurface(56*2, 64*2, 8*2, 8*2)
        brick1 = bricks.subsurface((0, 0, 8, 8))
        brick2 = bricks.subsurface((8, 0, 8, 8))
        brick3 = bricks.subsurface((8, 8, 8, 8))
        brick4 = bricks.subsurface((0, 8, 8, 8))

        alphabet = {
            "a": "0071b63c7ff1e3",
            "b": "01fb1e3fd8f1fe",
            "c": "00799e0c18199e",
            "e": "01fb060f98307e",
            "g": "007d860cf8d99f",
            "i": "01f8c183060c7e",
            "l": "0183060c18307e",
            "m": "018fbffffaf1e3",
            "o": "00fb1e3c78f1be",
            "r": "01fb1e3cff3767",
            "t": "01f8c183060c18",
            "v": "018f1e3eef8e08",
            "y": "019b3667860c18"
        }

        abs_x, abs_y = pos

        for letter in text.lower():

            binstr = ""
            for h in self.chunks(alphabet[letter], 2):
                binstr += str(bin(int(h, 16)))[2:].rjust(8, "0")
            binstr = binstr[7:]

            x, y = 0, 0
            letter_w = 0
            surf_letter = pygame.Surface((56, 56))
            for j, row in enumerate(self.chunks(binstr, 7)):
                for i, bit in enumerate(row):
                    if bit == "1":
                        if i%2 == 0 and j%2 == 0:
                            surf_letter.blit(brick1, [x, y])
                        elif i%2 == 1 and j%2 == 0:
                            surf_letter.blit(brick2, [x, y])
                        elif i%2 == 1 and j%2 == 1:
                            surf_letter.blit(brick3, [x, y])
                        elif i%2 == 0 and j%2 == 1:
                            surf_letter.blit(brick4, [x, y])
                        if x > letter_w:
                            letter_w = x
                    x += 8
                x = 0
                y += 8
            screen.blit(surf_letter, [abs_x, abs_y])
            abs_x += letter_w + 16

    def toggleEnemyFreeze(self, freeze=True):
        """
            Замораживает или размораживает всех врагов.
            :param freeze: True, чтобы заморозить; False, чтобы разморозить
        """

        global enemies

        for enemy in enemies:
            enemy.paused = freeze
        self.timefreeze = freeze


    def loadHiscore(self):
        """
            Загружает рекордный счет.
            Если не удается загрузить рекорд, возвращает 20000.
            :return: рекордный счет (int)
        """
        filename = ".hiscore"
        if (not os.path.isfile(filename)):
            return 20000

        f = open(filename, "r")
        hiscore = int(f.read())

        if hiscore > 19999 and hiscore < 1000000:
            return hiscore
        else:

            return 20000

    def saveHiscore(self, hiscore):
        """
            Сохраняет рекордный счет.
            :param hiscore: рекордный счет
            :return: True, если успешно; False, если ошибка
        """
        try:
            f = open(".hiscore", "w")
        except:

            return False
        f.write(str(hiscore))
        f.close()
        return True


    def finishLevel(self):
        """
            Завершает текущий уровень.
            Показывает заработанные очки и переходит на следующий уровень.
        """

        global play_sounds, sounds

        for player in players:
            player.lives = 3  # Сброс количества жизней игрока

        if play_sounds:
            sounds["bg"].stop()

        self.active = False
        if self.game_over:
            game.showMenu()
        else:
            self.nextLevel()
        print("Уровень "+str(self.stage)+" завершен")

    def nextLevel(self):
        """
            Запускает следующий уровень игры.
        """

        global castle, players, bullets, bonuses, play_sounds, sounds, screen_array, screen_array_grayscale
        # Очистка объектов предыдущего уровня
        del bullets[:]
        del enemies[:]
        del bonuses[:]
        castle.rebuild()
        del gtimer.timers[:]

        # Загрузка рандомного уровня
        self.stage = random.randint(1, 2)
        self.level = Level(self.stage)
        self.timefreeze = False

        # Генерация врагов
        self.level.enemies_left = [0] * 15

        if play_sounds:
            sounds["start"].play()
            gtimer.add(4330, lambda: sounds["bg"].play(-1), 1)

        """
            Случайная инициализация доступных позиций для размещения объектов на уровне.
            Читает файл уровня, анализирует доступные участки и добавляет их в список.
        """
        self.available_positions = []
        filename = "levels/gameplay/" + str(self.stage)
        f = open(filename, "r")
        data = f.read().split("\n")
        f.close()
        for y in range(len(data) - 1):
            row = data[y]
            for x in range(len(row) - 1): # Проверка на доступность позиции и её соответствие ограничениям
                if row[x] == "." and row[x + 1] == "." and data[y + 1][x] == "." and data[y+1][x+1] == "." and (not (x == 12 and y == 24)) and y > 12 and 3 < x < 21:
                    self.available_positions.append([x, y])
        random.shuffle(self.available_positions)

        # Перезагрузка игроков.
        # Возвращает игроков на стартовые позиции и обновляет их состояние.
        self.reloadPlayers()
        # Устанавливает таймер для создания новых врагов
        gtimer.add(2500, lambda: self.spawnEnemy())  # original was 5000,

        # Флаги состояния игры
        self.game_over = False  # Указатель на завершение игры

        # Если False, игра завершится
        self.running = True

        # Если False, игроки не смогут взаимодействовать
        self.active = True

        # Отрисовка начального состояния
        self.draw()

        screen_array = pygame.surfarray.array3d(screen)
        screen_array = np.transpose(screen_array, (1, 0, 2))
        screen_array_grayscale = rgb_to_grayscale(screen_array)

        # Инициализация ИИ-бота для обучения.
        self.agent = ai_agent()
        self.p_mapinfo = multiprocessing.Queue()  # Очередь для передачи информации о карте
        self.c_control = multiprocessing.Queue()  # Очередь для получения действий от ИИ

        # Получение информации о текущей карте
        mapinfo = self.get_mapinfo()
        self.agent.mapinfo = mapinfo
        if self.p_mapinfo.empty() == True:
            self.p_mapinfo.put(mapinfo)

        self.ai_bot_actions = [0, 4]
        self.p = multiprocessing.Process(target=self.agent.operations, args=(self.p_mapinfo, self.c_control))
        self.p.start()


    def get_mapinfo(self):
        global players, bullets
        mapinfo=[]
        mapinfo.append([])
        mapinfo.append([])
        mapinfo.append([])
        mapinfo.append([])
        for bullet in bullets:
            if bullet.owner == bullet.OWNER_ENEMY:
                nrect=bullet.rect.copy()
                mapinfo[0].append([nrect, bullet.direction, bullet.speed])
        for enemy in enemies:
            nrect=enemy.rect.copy()
            mapinfo[1].append([nrect, enemy.direction, enemy.speed, enemy.type])
        for tile in game.level.mapr:
            nrect=pygame.Rect(tile.left, tile.top, 16, 16)
            mapinfo[2].append([nrect, tile.type])
        for player in players:
            nrect=player.rect.copy()
            mapinfo[3].append([nrect, player.direction, player.speed, player.shielded])
        return mapinfo


'''
===============================================================================================================================
                                                        RL СРЕДА ОБУЧЕНИЯ
===============================================================================================================================
'''


class TanksEnv(gymnasium.Env):
    """
        Класс среды для игры в танки, интегрированный с OpenAI Gym.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        """
            Инициализация среды.

            :param render_mode: Режим отображения ("human" или "rgb_array").
        """
        # Определение глобальных переменных игры
        global gtimer, sprites, screen, screen_array, screen_array_grayscale, players, enemies, bullets, bonuses, labels, play_sounds, sounds, game, castle
        global obs_flag_castle_danger, obs_flag_stupid, obs_flag_player_collision, obs_flag_hot, obs_flag_bullet_fired
        # Флаги наблюдения
        obs_flag_castle_danger = 0
        obs_flag_stupid = 0
        obs_flag_player_collision = 0
        obs_flag_hot = 0
        obs_flag_bullet_fired = 0
        self.enemy_in_line = 4
        # Настройки экрана и карты
        self.width = 208   # ширина экрана
        self.height = 208  # высота экрана
        self.paso = 0

        # Initialize the heat map
        self.heat_map = np.zeros((13, 13))
        self.grid_size = 32
        self.grid_position = [0, 0, 0, 0, 0, 0, 0]
        self.heat_decay_rate = 0.02  ## скорость затухания тепловой карты
        self.heat_base_penalty = 0.01  # базовый штраф за нахождение в "горячей зоне"

        # Позиции врагов и пуль
        self.enemy_positions = np.full((4, 7), 0)
        self.bullet_positions = np.full((6, 7), 0)

        # Инициализация стека кадров
        self.frame_stack = deque(maxlen=4)
        empty_frame = np.zeros((self.width, self.height), dtype=np.uint8)
        for _ in range(4):
            self.frame_stack.append(empty_frame)

        self.bullet_avoidance_dir = 4

        # Пространство наблюдений
        self.observation_space = gymnasium.spaces.Dict(
            {
                "obs_frames": gymnasium.spaces.Box(low=0.0, high=1.0, shape=(4, self.width, self.height), dtype=np.float64),

                "player_position": gymnasium.spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float64),  # Normalized values
                "enemy1_position": gymnasium.spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float64),
                "enemy2_position": gymnasium.spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float64),
                "enemy3_position": gymnasium.spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float64),
                "enemy4_position": gymnasium.spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float64),

                "bullet1_position": gymnasium.spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float64),
                "bullet2_position": gymnasium.spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float64),
                "bullet3_position": gymnasium.spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float64),
                "bullet4_position": gymnasium.spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float64),
                "bullet5_position": gymnasium.spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float64),
                "bullet6_position": gymnasium.spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float64),

                "prev_action": gymnasium.spaces.MultiDiscrete([2, 5]),  # [no shoot, shoot], [move up, down, right , left, no move]
                "ai_bot_actions": gymnasium.spaces.MultiDiscrete([2, 5]),
                "flags": gymnasium.spaces.MultiBinary(5),
                "enemies_left": gymnasium.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64),
                "heatmap": gymnasium.spaces.Box(low=0.0, high=1.0, shape=(13, 13), dtype=np.float64),
            }
        )

        # Пространство действий
        self.action_space = gymnasium.spaces.MultiDiscrete([2, 5])

        # Инициализация игровых объектов
        gtimer = Timer()

        sprites = None
        screen = None
        screen_array = None
        screen_array_grayscale = empty_frame
        players = []
        enemies = []
        bullets = []
        bonuses = []
        labels = []

        play_sounds = False
        sounds = {}

        game = Game()
        castle = Castle()
        game.showMenu()

    def _get_obs(self):
        """
            Получение текущего состояния игры.

            :return: Словарь наблюдений.
        """
        global gtimer, sprites, screen, screen_array, screen_array_grayscale, players, enemies, bullets, bonuses, labels, play_sounds, sounds, game, castle
        global obs_flag_castle_danger, obs_flag_stupid, obs_flag_player_collision, obs_flag_hot, obs_flag_bullet_fired

        return {
            "obs_frames": np.array(self.obs_frames) / 255.0,

            "player_position": np.array(self.grid_position) / np.array([29*16, 29*16, 4, 4, 59*16, 59*16, 4]),
            "enemy1_position": np.array(self.enemy_positions[0]) / np.array([29*16, 29*16, 4, 1, 59*16, 59*16, 4]),
            "enemy2_position": np.array(self.enemy_positions[1]) / np.array([29*16, 29*16, 4, 1, 59*16, 59*16, 4]),
            "enemy3_position": np.array(self.enemy_positions[2]) / np.array([29*16, 29*16, 4, 1, 59*16, 59*16, 4]),
            "enemy4_position": np.array(self.enemy_positions[3]) / np.array([29*16, 29*16, 4, 1, 59*16, 59*16, 4]),

            "bullet1_position": np.array(self.bullet_positions[0]) / np.array([29*16, 29*16, 4, 1, 59*16, 59*16, 4]),
            "bullet2_position": np.array(self.bullet_positions[1]) / np.array([29*16, 29*16, 4, 1, 59*16, 59*16, 4]),
            "bullet3_position": np.array(self.bullet_positions[2]) / np.array([29*16, 29*16, 4, 1, 59*16, 59*16, 4]),
            "bullet4_position": np.array(self.bullet_positions[3]) / np.array([29*16, 29*16, 4, 1, 59*16, 59*16, 4]),
            "bullet5_position": np.array(self.bullet_positions[4]) / np.array([29*16, 29*16, 4, 1, 59*16, 59*16, 4]),
            "bullet6_position": np.array(self.bullet_positions[5]) / np.array([29*16, 29*16, 4, 1, 59*16, 59*16, 4]),

            "ai_bot_actions": np.array(game.ai_bot_actions),
            "prev_action": np.array(self.prev_action),

            "flags": np.array([obs_flag_castle_danger, obs_flag_stupid, obs_flag_player_collision, obs_flag_hot, obs_flag_bullet_fired]),
            "enemies_left": np.array([len(enemies) / 20]),
            "heatmap": np.array(self.heat_map) / 25,

        }

    def _get_info(self):
        """
            Возвращает дополнительную информацию о текущем состоянии игры.

            :return: Словарь с информацией.
        """
        return {"Info": 0}


    def kill_ai_process(self, p):
        """
            Убивает процесс AI.

            :param p: Процесс, который нужно завершить.
        """
        os.kill(p.pid, 9)
        #print("Killed AI Process!")

    def clear_queue(self, queue):
        """
            Очищает очередь.

            :param queue: Очередь, которую нужно очистить.
        """

        while not queue.empty():
            try:
                queue.get(False)
            except Empty:
                break

    def reset(self, seed=None, options=None):
        """
            Сбрасывает среду к начальному состоянию.

            :param seed: Сид для генерации случайных чисел (опционально).
            :param options: Дополнительные параметры (опционально).
            :return: Наблюдение и информация о состоянии.
        """
        global gtimer, sprites, screen, screen_array, screen_array_grayscale, players, enemies, bullets, bonuses, labels, play_sounds, sounds, game, castle
        global obs_flag_castle_danger, obs_flag_stupid, obs_flag_player_collision, obs_flag_hot, obs_flag_bullet_fired

        self.reward = 0
        self.paso = 0
        players[0].lives = 3
        self.prev_action = np.array([0, 0])

        self.bullet_avoidance_dir = 4
        obs_flag_castle_danger = 0
        obs_flag_stupid = 0
        obs_flag_player_collision = 0
        obs_flag_hot = 0
        obs_flag_bullet_fired = 0


        empty_frame = np.zeros((self.width, self.height), dtype=np.uint8)
        # Сбрасываем стек кадров
        self.frame_stack.clear()
        for _ in range(4):
            self.frame_stack.append(empty_frame)

        self.obs_frames = np.stack([empty_frame, empty_frame, empty_frame, empty_frame], axis=0)
        self.heat_map = np.zeros((13, 13))




        for i in range(4):
            self.enemy_positions[i] = [29*16, 29*16, 4, 0, 59*16, 59*16, 4]  # Dead

        for i in range(6):
            self.bullet_positions[i] = [29*16, 29*16, 4, 1, 59*16, 59*16, 4]  # Dead

        #game.gameOver()
        game.nextLevel()
        #game.finishLevel()
        game.ai_bot_actions = [0 if x is None else x for x in game.ai_bot_actions]
        observation = self._get_obs()
        info = self._get_info()

        return observation, info


    def step(self, action):
        """
            Выполняет один шаг симуляции.

            :param action: Действие, которое нужно выполнить.
            :return: Наблюдение, награда, завершен ли эпизод, информация.
        """
        global gtimer, sprites, screen, screen_array, screen_array_grayscale, players, enemies, bullets, bonuses, labels, play_sounds, sounds, game, castle
        global obs_flag_stupid, obs_flag_player_collision, obs_flag_hot, obs_flag_castle_danger, obs_flag_bullet_fired
        self.reward = 0
        obs_flag_stupid = 0
        obs_flag_player_collision = 0
        obs_flag_castle_danger = 0
        obs_flag_bullet_fired = 0
        self.bullet_avoidance_dir = 4
        time_passed = 20
        self.paso += 1

        # Обновляем позиции врагов
        for i, enemy in enumerate(enemies[:4]):
            if enemy.state != enemy.STATE_DEAD:

                grid_x, grid_y = enemy.rect.centerx, enemy.rect.centery
                direction = enemy.direction
                status = 1  # Живой
                distance_to_castle = Vmanhattan_distance(enemy.rect.topleft, castle.rect.topleft)
                distance_to_player = Vmanhattan_distance(enemy.rect.topleft, players[0].rect.topleft)
                in_line_status = Vinline_with_enemy(players[0].rect, enemy.rect)
            else:
                grid_x, grid_y, direction, status, distance_to_castle, distance_to_player, in_line_status = 29*16, 29*16, 4, 0, 59*16, 59*16, 4  # Dead

            self.enemy_positions[i] = [grid_x, grid_y, direction, status, distance_to_castle, distance_to_player, in_line_status]

        # Если врагов меньше 4, отметьте оставшиеся позиции как мертвые
        if len(enemies) < 4:
            for i in range(len(enemies), 4):
                self.enemy_positions[i] = [29*16, 29*16, 4, 0, 59*16, 59*16, 4]  # Dead status for non-existent enemies

        # Обновляем позиции пуль
        for i, bullet in enumerate(bullets[:6]):
            if bullet.state != bullet.STATE_REMOVED:
                # Преобразовать позицию пикселя в позицию сетки (НЕТ!)
                grid_x, grid_y = bullet.rect.centerx, bullet.rect.centery
                direction = bullet.direction
                owner = bullet.owner
                distance_to_castle = Vmanhattan_distance(bullet.rect.topleft, castle.rect.topleft)
                distance_to_player = Vmanhattan_distance(bullet.rect.topleft, players[0].rect.topleft)
                in_line_status = Vinline_with_enemy(players[0].rect, bullet.rect)
            else:
                grid_x, grid_y, direction, owner, distance_to_castle, distance_to_player, in_line_status = 29*16, 29*16, 4, 0, 59*16, 59*16, 4  # Dead

            self.bullet_positions[i] = [grid_x, grid_y, direction, owner, distance_to_castle, distance_to_player, in_line_status]

        # Если врагов меньше 4, отметьте оставшиеся позиции как мертвые
        if len(bullets) < 6:
            for i in range(len(bullets), 4):
                self.bullet_positions[i] = [29*16, 29*16, 4, 0, 59*16, 59*16, 4]  # Dead status for non-existent enemies


        # Инициализация минимального расстояния
        smallest_distance = 59*16

        for enemy in self.enemy_positions:
            status = enemy[3]
            distance_to_castle = enemy[4]
            distance_to_player = enemy[5]

            # Проверьте, жив ли противник, и при необходимости обновите наименьшее расстояние
            if status == 1 and distance_to_player < smallest_distance:
                smallest_distance = distance_to_player
            if status == 1 and distance_to_castle < 10*16:
                obs_flag_castle_danger = 1


        if len(bullets) != 0:
            bullets_info=[]
            bullets_info.append([])
            for bullet in bullets:
                if bullet.owner == bullet.OWNER_ENEMY:
                    nrect=bullet.rect.copy()
                    bullets_info[0].append([nrect,bullet.direction,bullet.speed])
            self.bullet_avoidance_dir = Vbullet_avoidance(players[0].rect, bullets_info[0])


        # Обновите информацию о карте и получите действия бота ИИ
        mapinfo = game.get_mapinfo()
        if game.p_mapinfo.empty() == True:
            game.p_mapinfo.put(mapinfo)
        if game.c_control.empty() != True:
            try:
                game.ai_bot_actions = game.c_control.get(False)
            except queue.empty:
                skip_this = True

        # Константы, представляющие направления
        DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT = range(4)
        pygame.event.pump()
        game.ai_bot_actions = [0.0 if x is None else x for x in game.ai_bot_actions]


        for player in players:
            if player.state == player.STATE_ALIVE and not game.game_over and game.active:
                if action[0] == 1 and not antiStupidBlock(player.direction, player.rect, castle.rect):
                    player.fire()

                if action[0] == 1 and antiStupidBlock(player.direction, player.rect, castle.rect):
                    self.reward -= 0.1

                if action[1] == 0:
                    player.move(game.DIR_UP)
                    if self.prev_action[1] == 0 and obs_flag_player_collision == 0:
                        self.reward += 0.05
                        pass

                if action[1] == 1:
                    player.move(game.DIR_RIGHT)
                    if self.prev_action[1] == 1 and obs_flag_player_collision == 0:
                        self.reward += 0.05
                        pass

                if action[1] == 2:
                    player.move(game.DIR_DOWN)
                    if self.prev_action[1] == 2 and obs_flag_player_collision == 0:
                        self.reward += 0.05
                        pass

                if action[1] == 3:
                    player.move(game.DIR_LEFT)
                    if self.prev_action[1] == 3 and obs_flag_player_collision == 0:
                        self.reward += 0.05
                        pass

            self.prev_action = action
            if action[0] == game.ai_bot_actions[0] and action[0] == 1:
                self.reward += 0.2
                pass

            if action[1] == game.ai_bot_actions[1] and action[1] != 4:
                self.reward += 0.1
                pass

            # Получить текущую позицию игрока на сетке
            distance_to_castle = Vmanhattan_distance(player.rect.topleft, castle.rect.topleft)
            self.grid_position = (player.rect.centerx, player.rect.centery, player.direction, player.lives, distance_to_castle, smallest_distance, self.bullet_avoidance_dir)

            # Увеличить температуру сгорания текущей позиции
            if self.heat_map[round(self.grid_position[0]//self.grid_size), round(self.grid_position[1]//self.grid_size)] < 25:
                self.heat_map[round(self.grid_position[0]//self.grid_size), round(self.grid_position[1]//self.grid_size)] += 0.5

            if self.heat_map[round(self.grid_position[0]//self.grid_size), round(self.grid_position[1]//self.grid_size)]	> 9:
                obs_flag_hot = 1
            else:
                obs_flag_hot = 0

            # Применить отрицательное вознаграждение на основе тепловой ценности
            self.reward -= self.heat_base_penalty * (1.22 ** self.heat_map[round(self.grid_position[0]//self.grid_size), round(self.grid_position[1]//self.grid_size)])

            # Уменьшить тепловую карту
            self.heat_map *= (1 - self.heat_decay_rate)
            #print(np.round(self.heat_map, 1))

            player.update(time_passed)

        for enemy in enemies:

            if enemy.state == enemy.STATE_DEAD and not game.game_over and game.active:
                self.reward += 5  # RW KILL
                if enemy.rect.y > 208:
                    self.reward += 10  # RW KILL ENEMY CLOSE TO BASE
                #self.killed_enemies += 1

                enemies.remove(enemy)

                if len(game.level.enemies_left) == 0 and len(enemies) == 0:
                    self.reward += 50  # RW WIN
                    print("Вы уничтожили все вражеские танки.! :)")
                    self.kill_ai_process(game.p)
                    self.clear_queue(game.p_mapinfo)
                    self.clear_queue(game.c_control)
                    game.game_over = 1
            else:
                enemy.update(time_passed)

        if not game.game_over and game.active:
            for player in players:
                if player.state == player.STATE_ALIVE:
                    if player.bonus != None and player.side == player.SIDE_PLAYER:
                        game.triggerBonus(player.bonus, player)
                        self.reward += 1 # RW BONUS
                        player.bonus = None
                elif player.state == player.STATE_DEAD:
                    self.reward -= 5 # RW DEAD
                    #print("-50 for dying! ", self.reward)
                    game.superpowers = 0
                    player.lives -= 1
                    if player.lives > 0:
                        game.respawnPlayer(player)
                    else:
                        player.lives = 0
                        self.reward -= 15
                        print("Ты умер! :(")
                        self.kill_ai_process(game.p)
                        self.clear_queue(game.p_mapinfo)
                        self.clear_queue(game.c_control)
                        game.game_over = 1

        for bullet in bullets:

            if bullet.state == bullet.STATE_REMOVED:
                bullets.remove(bullet)
            else:
                bullet.update()
                if bullet.state == bullet.STATE_REMOVED:
                    bullets.remove(bullet)
                else:
                    bullet.update()
                    if bullet.state == bullet.STATE_REMOVED:
                        bullets.remove(bullet)
                    else:
                        bullet.update()
                        if bullet.state == bullet.STATE_REMOVED:
                            bullets.remove(bullet)
                        else:
                            bullet.update()
                            if bullet.state == bullet.STATE_REMOVED:
                                bullets.remove(bullet)
                            else:
                                bullet.update()

        for bullet in bullets:
            if bullet.owner == 0 and bullet.state == bullet.STATE_ACTIVE:
                    obs_flag_bullet_fired = 1


            if bullet.owner == Bullet.OWNER_PLAYER:
                # Проверяем, движется ли пуля в направлении базы
                if bullet.direction == DIR_DOWN and bullet.rect.bottom < castle.rect.top and bullet.rect.left <= castle.rect.right and bullet.rect.right >= castle.rect.left:
                    obs_flag_stupid = 1
                if bullet.direction == DIR_UP and bullet.rect.top > castle.rect.bottom and bullet.rect.left <= castle.rect.right and bullet.rect.right >= castle.rect.left:
                    obs_flag_stupid = 1
                if bullet.direction == DIR_RIGHT and bullet.rect.right < castle.rect.left and bullet.rect.top <= castle.rect.bottom and bullet.rect.bottom >= castle.rect.top:
                    obs_flag_stupid = 1
                if bullet.direction == DIR_LEFT and bullet.rect.left > castle.rect.right and bullet.rect.top <= castle.rect.bottom and bullet.rect.bottom >= castle.rect.top:
                    obs_flag_stupid = 1

        if obs_flag_stupid == 1:
            self.reward -= 0.1
            pass



        for bonus in bonuses:
            if bonus.active == False:
                bonuses.remove(bonus)

        for label in labels:
            if not label.active:
                labels.remove(label)

        if not game.game_over:
            if not castle.active:
                self.reward -= 50  # RW LOST
                print("База разрушена!")
                self.kill_ai_process(game.p)
                self.clear_queue(game.p_mapinfo)
                self.clear_queue(game.c_control)
                game.game_over = 1

        gtimer.update(time_passed)


        game.draw()  #  RENDER

        # Обновляем наблюдение новым текущим кадром
        screen_array = pygame.surfarray.array3d(screen)
        screen_array = np.transpose(screen_array, (1, 0, 2))
        screen_array_grayscale = rgb_to_grayscale(screen_array)

        # Add the new frame to the stack and update the specific frames
        self.frame_stack.append(screen_array_grayscale)

        self.obs_frames = np.stack([screen_array_grayscale, self.frame_stack[-2], self.frame_stack[-3], self.frame_stack[-4]], axis=0)

        observation = self._get_obs()

        #self.reward += self.killed_enemies*0.001
        reward = self.reward / 10
        terminated = game.game_over
        truncated = False
        # if self.paso == 2999 and terminated != 1:
        # 	truncated = True
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def _render_frame(self):
        pass

    def close(self):
        pass