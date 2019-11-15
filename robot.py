import numpy as np
from occupancy_grid import OccupancyGrid


class Robot(object):

    def __init__(self, name, gr, coord):
        self.pos = coord
        self.map = gr
        self.name = name
        self.map[coord[0]][coord[1]] = self.name

    def move_right(self):
        self.map[self.pos[0]][self.pos[1]] = 0
        self.pos[1] = self.pos[1] + 1 if self.pos[1] < len(self.map[0]) - 1 else self.pos[1]
        self.map[self.pos[0]][self.pos[1]] = self.name

    def move_left(self):
        self.map[self.pos[0]][self.pos[1]] = 0
        self.pos[1] = self.pos[1] - 1 if self.pos[1] > 0 else self.pos[1]
        self.map[self.pos[0]][self.pos[1]] = self.name

    def move_up(self):
        self.map[self.pos[0]][self.pos[1]] = 0
        self.pos[0] = self.pos[0] - 1 if self.pos[0] > 0 else self.pos[0]
        self.map[self.pos[0]][self.pos[1]] = self.name

    def move_down(self):
        self.map[self.pos[0]][self.pos[1]] = 0
        self.pos[0] = self.pos[0] + 1 if self.pos[0] < len(self.map) - 1 else self.pos[0]
        self.map[self.pos[0]][self.pos[1]] = self.name

    def sense(self):
        l = []
        for i in range(-1, 1):
            for j in range(-1, 1):
                if (i == j and j == 0):
                    continue
                posx = self.pos[0] + j
                posy = self.pos[1] + i
                if ((posx >= 0 and posx < len(self.map[0])) and (posy >= 0 and posy < len(self.map))):
                    l.append(self.map[posx][posy])
                else:
                    l.append(-1)
        return l

    def get_position(self):
        return self.pos


if __name__ == "__main__":
    og = Robot()
    og.show()
