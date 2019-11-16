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
        self.pos[1] = self.pos[1] + 1 if self.pos[1] < self.limits[1] - 1 else self.pos[1]
        return self.pos

    def move_left(self):
        self.pos[1] = self.pos[1] - 1 if self.pos[1] > 0 else self.pos[1]
        return self.pos

    def move_up(self):
        self.pos[0] = self.pos[0] - 1 if self.pos[0] > 0 else self.pos[0]
        return self.pos

    def move_down(self):
        self.pos[0] = self.pos[0] + 1 if self.pos[0] < self.limits[0] - 1 else self.pos[0]
        return self.pos

    def setObs(self, observation):
        self.obs = observation #3*3
        list_global = {}
        for i in range(3):
            for j in range(3):
                if observation[i][j] == -1:
                    continue
                observation[i][j] = 0 if observation[i][j] >= 3 else observation[i][j]
                gx = self.pos[0]
                gy = self.pos[1]
                lx, ly = 1, 1
                nx = gx - (lx - i)
                ny = gy - (ly - j)
                list_global[(nx, ny)] = observation[i][j]
        for pose, obs in list_global.items():
            # print(pose, obs)
            # print(self.get_static_inv_sensor_model(pose, obs))
            self.update_static_grid(self.get_static_inv_sensor_model(pose, obs), pose)


 
    def get_position(self):
        return self.pos

    def get_static_inv_sensor_model(self, pose, obs):
        """
        :param s_prob_val: The previous occupancy probability value at the (x,y) position
        :param obs: The observation at a particular (x,y) location. 0: Free | 1: Occupied
        :return: The inverse sensor model
        """
        s_prob_val = self.S.get_arr()[pose]
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

    def get_dynamic_inv_sensor_model(self, pose, obs):
        """
        :param s_prob_val: The previous occupancy probability value at the (x,y) position
        :param obs: The observation at a particular (x,y) location. 0: Free | 1: Occupied
        :return: The inverse sensor model
        """
        s_prob_val = self.S.get_arr()[pose]
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

    def update_static_grid(self, inv_sensor_model, pose):
        prev_S = self.S.get_arr()[pose]
        S_update = 0
        for x, y in np.ndindex(self.S.get_arr().shape):
            c = np.exp(np.log(inv_sensor_model / (1 - inv_sensor_model)) + np.log(prev_S / (1 - prev_S)))
            S_update = c/(1 + c)
        self.S.update(S_update, pose)

    def update_dynamic_grid(self, inv_sensor_model, pose):
        prev_D = self.D.get_arr()[pose]
        D_update = 0
        for x, y in np.ndindex(self.D.get_arr().shape):
            c = np.exp(np.log(inv_sensor_model / (1 - inv_sensor_model)) + np.log(prev_D / (1 - prev_D)))
            D_update[x, y] = c / (1 + c)
        self.D.update(D_update, pose)


    def step(self,action):
        self.time +=1
        if action == "r":
            return self.move_right()
        elif action == "o":
            return self.get_position()
        elif action == "l":
            return  self.move_left()
        elif action == "u":
            return  self.move_up()
        elif action == "d":
            return  self.move_down()

if __name__ == "__main__":
    og = Robot()
    og.show()
