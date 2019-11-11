import numpy as np
import pygame

class OccupancyGrid(object):
	
	def __init__(self, modes=True, length=4, width=4):
		self.mat = np.random.uniform(0.4,0.6, (length,width))
		if modes:
			self.mat = self.mat > 0.5
			self.mat = self.mat.astype(int)
		self.colors = np.array([[120, 250, 90], [250, 90, 120], [255, 255, 255]])

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
	og.show()
