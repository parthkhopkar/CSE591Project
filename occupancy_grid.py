import numpy as np
import pygame

class OccupancyGrid(object):
    def __init__(self, mode=True, length=4, width=4):
        if not mode:
            self.mat = np.random.uniform(0.5,0.5, (length,width))
        if mode:
            self.mat = np.zeros((length,width), np.int)
        self.colors = np.array([[255, 255, 255], [250, 90, 120], [120, 250, 90]])
    
    def add_static_object(self,posx,posy):
        self.mat[posx][posy] = 1
    
    def add_dynamic_object(self,posx,posy):
        self.mat[posx][posy] = 2
	
    def show(self):
        print(self.mat)
        pygame.init()
        screen = pygame.display.set_mode((640, 480))
        surface = pygame.pixelcopy.make_surface(self.colors[self.mat])
        surface = pygame.transform.scale(surface, (200, 200))
        screen.fill((30, 30, 30))
        screen.blit(surface, (100, 100))
        pygame.display.flip()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

		
if __name__ == "__main__":
    og = OccupancyGrid()
    og.add_static_object(0,3)
    og.add_static_object(1,2)
    og.add_static_object(1,0)
    og.add_dynamic_object(0,0)
    og.show()
