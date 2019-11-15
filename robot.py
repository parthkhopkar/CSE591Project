import numpy as np
from occupancy_grid import OccupancyGrid


class Robot(object):

    def __init__(self, name, coord, dims):
		self.time = 0
        self.pos = coord
        self.name = name
        self.limits = dims
		self.obs = None                                                       
		self.S = OccupancyGrid(False)                                                   
		self.D = OccupancyGrid(False)                                                      

    def move_right(self):
		## check if no static  object
        self.pos[1] = self.pos[1] + 1 if self.pos[1] < dims[1] - 1 else self.pos[1]
        return self.pos

    def move_left(self):
        self.pos[1] = self.pos[1] - 1 if self.pos[1] > 0 else self.pos[1]
        return self.pos

    def move_up(self):
        self.pos[0] = self.pos[0] - 1 if self.pos[0] > 0 else self.pos[0]
		return self.pos

    def move_down(self):
        self.pos[0] = self.pos[0] + 1 if self.pos[0] < dims[0] - 1 else self.pos[0]
		return self.pos

	def setObs(self, observation):
		self.obs = observation
 
    def get_position(self):
        return self.pos
	
	def __get_static_inv_sensor_model(self, s_prob_val, obs):
		"""
		:param s_prob_val: The previous occupancy probability value at the (x,y) position
		:param obs: The observation at a particular (x,y) location. 0: Free | 1: Occupied
		:return: The inverse sensor model
		"""
		free_threshold = 0.2  # Probability below which position is certainly free
		occupied_threshold = 0.8  # Probability above which position is certainly occupied
		inv_sensor_low = 0.1
		inv_sensor_high = 0.9
		if s_prob_val <= free_threshold:
			return inv_sensor_low  # Free, Free and Free, Occupied
		elif s_prob_val >= occupied_threshold:
			if obs:
				return inv_sensor_high  # Occupied, Occupied
			else:
				return inv_sensor_low  # Occupied, Free
		else:  # If unknown
			if obs:
				return inv_sensor_high  # Unknown, Occupied
			else:
				return inv_sensor_low  # Unknown, Free

	def __get_dynamic_inv_sensor_model(s_prob_val, obs):
		"""
		:param s_prob_val: The previous occupancy probability value at the (x,y) position
		:param obs: The observation at a particular (x,y) location. 0: Free | 1: Occupied
		:return: The inverse sensor model
		"""
		free_threshold = 0.2  # Probability below which position is certainly free
		occupied_threshold = 0.8  # Probability above which position is certainly occupied
		inv_sensor_low = 0.1
		inv_sensor_high = 0.9
		if s_prob_val <= free_threshold:
			if obs:
				return inv_sensor_high  # Free, Occupied
			else:
				return  inv_sensor_low  # Free, Free
		elif s_prob_val >= occupied_threshold:
			return inv_sensor_low  # Occupied, Free and Occupied, Occupied
		else:  # If unknown
			return inv_sensor_low  # Unknown, Free and Unknown, Occupied

	def update_static_grid(self, inv_sensor_model):
		prev_S = self.S.get_arr().copy()
		S_update = np.zeros(self.limits)
		for x, y in np.ndindex(self.limits):
			c = math.e ** math.log(inv_sensor_model / (1 - inv_sensor_model)) + math.log(prev_S[x, y] / (1 - prev_S[x, y]))
			S_update[x, y] = c/(1 + c)
		self.S.update(S_update)

    def update_dynamic_grid(self, inv_sensor_model):
        prev_D = self.D.get_arr().copy()
        D_update = np.zeros(self.limits)
        for x, y in np.ndindex(self.limits):
            c = math.e ** math.log(inv_sensor_model / (1 - inv_sensor_model)) + math.log(prev_D[x, y] / (1 - prev_D[x, y]))
            D_update[x, y] = c / (1 + c)
        self.D.update(D_update)

	
	def step(self,action):
		self.time +=1
		if action == "left":
			return self.move_right()
		elif action == "observe":
			return self.get_position()
		elif action == "right":
			return  self.move_left()
		elif action == "up":
			return  self.move_up()
		elif action == "down":
			return  self.move_down()

if __name__ == "__main__":
    og = Robot()
    og.show()
