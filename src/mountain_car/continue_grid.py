import numpy as np
import random

class Continue_grid:

    """
    un tableau représentant un plan continue
    les coordonées sont des flotants
    les cases du tableaux peuvent être raffiné pour obtenir plus de précision.
    Une case de la grille peut sois contenir une valeur, sois une nouvelle grille de taille x_step, y_step discretisant les valeur de la case avec plus de précision.
    """
    
    def __init__(self, x_min, x_max, y_min, y_max, x_step, y_step, init_val):
        """
        :param i_step: pas de discretisation de la grille celon l'axe i
        """
        
        x_len = abs(x_min) + abs(x_max)
        y_len = abs(y_min) + abs(y_max)
        self.i = 0
        self.grid = [[(False, init_val) for _ in range(x_step)] for _ in range(y_step)]

        # for i in range(y_step):
        #     for j in range(x_step):
        #         self.grid[i][j] = (False, (random.randint(-10, 0), 0, 0))
        
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.x_step = x_len / x_step
        self.y_step = y_len / y_step



    def __iter__(self):
        return Grid_iterator(self)
        
    def get(self, x, y):
        x_coord = int((x +abs(self.x_min))/ self.x_step)
        y_coord = int((y +abs(self.y_min))/ self.y_step)
        (is_grid, val) = self.grid[x_coord][y_coord]

        if not is_grid:
            return val
        else:
            return val.get(x % self.x_step, y % self.y_step)

    def set(self, x, y, new_val):
        x_coord = int((x +abs(self.x_min))/ self.x_step)
        y_coord = int((y +abs(self.y_min))/ self.y_step)
        (is_grid, val) = self.grid[x_coord][y_coord]

        if not is_grid:
            self.grid[x_coord][y_coord] = (is_grid, new_val)
        else:
            return val.set(x % self.x_step, y % self.y_step, new_val)

    def discretise(self, x, y, x_step, y_step):
        x_coord = int((x +abs(self.x_min))/ self.x_step)
        y_coord = int((y +abs(self.y_min))/ self.y_step)
        (is_grid, val) = self.grid[x_coord][y_coord]

        if not is_grid:
            new_grid = Continue_grid(0, self.x_step, 0,  self.y_step,
                                     x_step, y_step, val)
            self.grid[x_coord][y_coord] = (True, new_grid)
        else:
            return val.discretise(x % self.x_step, y % self.y_step,
                                  x_step, y_step)

class Grid_iterator:

    def __init__(self, grid):
        # self.grid = grid
        self.stack = [(0, 0, grid)]
        # self.i = 0
        # self.j = 0

    def __next__(self):
        (i, j, grid) = self.stack.pop()
        if i == len(grid.grid[0]):
            i = 0
            j += 1
        if j == len(grid.grid):
            if self.stack == []:
                raise StopIteration
            else:
                return self.__next__()
        
        is_grid, val = grid.grid[j][i]
        if is_grid:
            self.stack.append((i + 1, j, grid))
            self.stack.append((0, 0, val))
            return self.__next__()
        else:
            self.stack.append((i + 1, j, grid))
            return val

