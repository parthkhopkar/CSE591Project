import numpy as np
import pygame
import time


class OccupancyGrid(object):
    def __init__(self, mode=True, length=4, width=4):
        if not mode:
            self.mat = np.full((length, width), 0.5)
        if mode:
            self.mat = np.zeros((length, width), np.int)
        self.colors = np.array([[255, 255, 255], [250, 90, 120], [0, 255, 0], [0, 0, 0], [0, 0, 0]])
        self.nd = 0
        self.dot = {}
        self._image_num = 0

    def add_static_object(self,posx, posy):
        self.mat[posx][posy] = 1

    def add_dynamic_object(self, posx, posy, posx1, posy1):
        self.mat[posx][posy] = 2
        self.dot[self.nd] = ((posx, posy), (posx1, posy1))
        self.nd += 1

    def show(self, mode=True):
        print(self.mat)
        pygame.init()
        screen = pygame.display.set_mode((640, 480))
        if mode:
            surface = pygame.pixelcopy.make_surface(self.colors[self.mat])
        else:
            x = self.mat > 0.5
            x = np.array(x, np.int)
            surface = pygame.pixelcopy.make_surface(self.colors[x])
        surface = pygame.transform.scale(surface, (200, 200))
        screen.fill((30, 30, 30))
        screen.blit(surface, (100, 100))
        pygame.display.update()
        #time.sleep(1)
        self._image_num += 1
        str_num = "./images/000" + str(self._image_num)
        file_name = "image" + str_num[-4:] + ".jpg"
        pygame.image.save(screen, file_name)
        '''running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False'''

    def get_arr(self):
        return self.mat

    def update(self, new_val, pose):
        self.mat[pose] = new_val

    def update_robot_position(self, name, old_pose, new_pose):
        self.mat[old_pose[0]][old_pose[1]] = 0
        self.mat[new_pose[0]][new_pose[1]] = name

    def step(self):
        for i in range(self.nd):
            pose1, pose2 = self.dot[i]
            if self.mat[pose1[0]][pose1[1]] == 2:
                self.mat[pose1[0]][pose1[1]] = 0
                self.mat[pose2[0]][pose2[1]] = 2
            else:
                self.mat[pose1[0]][pose1[1]] = 2
                self.mat[pose2[0]][pose2[1]] = 0

if __name__ == "__main__":
    og = OccupancyGrid()
    og.add_static_object(0, 3)
    og.add_static_object(1, 2)
    og.add_static_object(1, 0)
    og.add_dynamic_object(0, 0)
    og.show()
