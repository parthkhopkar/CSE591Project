import numpy as np
from occupancy_grid import OccupancyGrid


class Robot(object):

    def __init__(self, name=0, coord=(1,1), dims=[3,3]):
        self.time = 0
        self.pos = coord
        self.name = name
        self.limits = dims
        self.obs = None
        self.S = OccupancyGrid(False, dims[0], dims[1])
        self.D = OccupancyGrid(False, dims[0], dims[1])
        self.T = OccupancyGrid(length=dims[0], width=dims[1])
        self.C = 80.0

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
            self.update_dynamic_grid(self.get_dynamic_inv_sensor_model(pose, obs), pose)
            self.T.update(self.time, pose)

        self.update_time()

	def ekf_slam(self, xEst, PEst, u, z, dObj):
		# Predict
		S = self.STATE_SIZE
		xEst[0:S] = self.motion_model(xEst[0:S], u)
		G, Fx = self.jacob_motion(xEst[0:S], u)
		PEst[0:S, 0:S] = G.T * PEst[0:S, 0:S] * G + Fx.T * Cx * Fx
		initP = np.eye(2)
		print(z)
		# Update
		for iz in range(len(z[:, 0])):  # for each observation
			min_id = self.search_correspond_landmark_id(xEst, PEst, z[iz, 0:2])
			
			if(z[iz, 2] in dObj):
				continue
				
			nLM = calc_n_lm(xEst)
			
			if min_id == nLM:
				#print("New LM")
				# Extend state and covariance matrix
				xAug = np.vstack((xEst, self.calc_landmark_position(xEst, z[iz, :])))
				PAug = np.vstack((np.hstack((PEst, np.zeros((len(xEst), LM_SIZE)))),
								  np.hstack((np.zeros((LM_SIZE, len(xEst))), initP))))
				xEst = xAug
				PEst = PAug
			lm = self.get_landmark_position_from_state(xEst, min_id)
				
			y, S, H = self.calc_innovation(lm, xEst, PEst, z[iz, 0:2], min_id)

			K = (PEst @ H.T) @ np.linalg.inv(S)
			xEst = xEst + (K @ y)
			PEst = (np.eye(len(xEst)) - (K @ H)) @ PEst

		xEst[2] = self.pi_2_pi(xEst[2])

		return xEst, PEst


	def calc_input(self):
		v = 1.0  # [m/s]
		yaw_rate = 0.1  # [rad/s]
		u = np.array([[v, yaw_rate]]).T
		return u

	def getDynamicObjects(self, time):
		if(time > 60):
			return [2,3]
		else:
			return []

	def observation(self, xTrue, xd, u, RFID, time):
		xTrue = motion_model(xTrue, u)

		# add noise to gps x-y
		z = np.zeros((0, 3))
		
		for i in range(len(RFID[:, 0])):

			dx = RFID[i, 0] - xTrue[0, 0]
			dy = RFID[i, 1] - xTrue[1, 0]
			d = math.sqrt(dx ** 2 + dy ** 2)
			angle = self.pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0])
			if d <= MAX_RANGE:
				dn = d + np.random.randn() * self.Q_sim[0, 0] ** 0.5  # add noise
				angle_n = angle + np.random.randn() * self.Q_sim[1, 1] ** 0.5  # add noise
				zi = np.array([dn, angle_n, i])
				z = np.vstack((z, zi))
		
		dObj = self.getDynamicObjects(time)

		# add noise to input
		ud = np.array([[
			u[0, 0] + np.random.randn() * R_sim[0, 0] ** 0.5,
			u[1, 0] + np.random.randn() * R_sim[1, 1] ** 0.5]]).T

		xd = self.motion_model(xd, ud)
		return xTrue, z, xd, ud, dObj


	def motion_model(self, x, u):
		F = np.array([[1.0, 0, 0],
					  [0, 1.0, 0],
					  [0, 0, 1.0]])

		B = np.array([[DT * math.cos(x[2, 0]), 0],
					  [DT * math.sin(x[2, 0]), 0],
					  [0.0, DT]])

		x = (F @ x) + (B @ u)
		return x


	def calc_n_lm(self, x):
		n = int((len(x) - STATE_SIZE) / LM_SIZE)
		return n


	def jacob_motion(self, x, u):
		Fx = np.hstack((np.eye(STATE_SIZE), np.zeros(
			(STATE_SIZE, LM_SIZE * calc_n_lm(x)))))

		jF = np.array([[0.0, 0.0, -DT * u[0] * math.sin(x[2, 0])],
					   [0.0, 0.0, DT * u[0] * math.cos(x[2, 0])],
					   [0.0, 0.0, 0.0]])

		G = np.eye(STATE_SIZE) + Fx.T * jF * Fx

		return G, Fx,


	def calc_landmark_position(self ,x, z):
		zp = np.zeros((2, 1))

		zp[0, 0] = x[0, 0] + z[0] * math.cos(x[2, 0] + z[1])
		zp[1, 0] = x[1, 0] + z[0] * math.sin(x[2, 0] + z[1])

		return zp


	def get_landmark_position_from_state(self, x, ind):
		lm = x[STATE_SIZE + LM_SIZE * ind: STATE_SIZE + LM_SIZE * (ind + 1), :]

		return lm


	def search_correspond_landmark_id(self, xAug, PAug, zi):
		"""
		Landmark association with Mahalanobis distance
		"""

		nLM = calc_n_lm(xAug)

		min_dist = []

		for i in range(nLM):
			lm = self.get_landmark_position_from_state(xAug, i)
			y, S, H = self.calc_innovation(lm, xAug, PAug, zi, i)
			min_dist.append(y.T @ np.linalg.inv(S) @ y)

		min_dist.append(M_DIST_TH)  # new landmark

		min_id = min_dist.index(min(min_dist))

		return min_id


	def calc_innovation(self, lm, xEst, PEst, z, LMid):
		delta = lm - xEst[0:2]
		q = (delta.T @ delta)[0, 0]
		z_angle = math.atan2(delta[1, 0], delta[0, 0]) - xEst[2, 0]
		zp = np.array([[math.sqrt(q), pi_2_pi(z_angle)]])
		y = (z - zp).T
		y[1] = self.pi_2_pi(y[1])
		H = jacob_h(q, delta, xEst, LMid + 1)
		S = H @ PEst @ H.T + Cx[0:2, 0:2]

		return y, S, H


	def jacob_h(self, q, delta, x, i):
		sq = math.sqrt(q)
		G = np.array([[-sq * delta[0, 0], - sq * delta[1, 0], 0, sq * delta[0, 0], sq * delta[1, 0]],
					  [delta[1, 0], - delta[0, 0], - q, - delta[1, 0], delta[0, 0]]])

		G = G / q
		nLM = self.calc_n_lm(x)
		F1 = np.hstack((np.eye(3), np.zeros((3, 2 * nLM))))
		F2 = np.hstack((np.zeros((2, 3)), np.zeros((2, 2 * (i - 1))),
						np.eye(2), np.zeros((2, 2 * nLM - 2 * i))))

		F = np.vstack((F1, F2))

		H = G @ F

		return H


	def pi_2_pi(self, angle):
		return (angle + math.pi) % (2 * math.pi) - math.pi

    def update_time(self):
        observed = np.array(self.T.get_arr() > 0, np.int)
        delT = np.full((self.limits[0], self.limits[1]), self.time) * observed - self.T.get_arr()
        delT = delT/self.C
        self.T.mat = self.T.mat - delT

    def merge(self, robot_S, robot_D, robot_T):
        for i in range(self.limits[0]):
            for j in range(self.limits[1]):
                if robot_T[i, j] > self.T.get_arr()[i, j]:
                    self.S.update(robot_S[i, j], (i, j))
                    self.D.update(robot_D[i, j], (i, j))
                    self.T.update(robot_T[i, j], (i, j))




    def get_position(self):
        return self.pos

    def get_static_inv_sensor_model(self, pose, obs):
        """
        :param s_prob_val: The previous occupancy probability value at the (x,y) position
        :param obs: The observation at a particular (x,y) location. 0: Free | 1: Occupied
        :return: The inverse sensor model
        """
        # print(self.S.get_arr())
        s_prob_val = self.S.get_arr()[pose]
        free_threshold = 0.45  # Probability below which position is certainly free
        occupied_threshold = 0.55  # Probability above which position is certainly occupied
        inv_sensor_low = 0.05
        inv_sensor_high = 0.85
        if s_prob_val <= free_threshold:
            if obs:
                return 0.01
            else:
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
        free_threshold = 0.1  # Probability below which position is certainly free
        occupied_threshold = 0.9  # Probability above which position is certainly occupied
        inv_sensor_low = 0.05
        inv_sensor_high = 0.99
        if s_prob_val <= free_threshold:
            if obs:
                return inv_sensor_high  # Free, Occupied
            else:
                return inv_sensor_low  # Free, Free
        elif s_prob_val >= occupied_threshold:
            if obs:
                return inv_sensor_low  # Occupied, Free and Occupied, Occupied
            else:
                return 0.2
        else:  # If unknown
            return inv_sensor_low  # Unknown, Free and Unknown, Occupied

    def update_static_grid(self, inv_sensor_model, pose):
        prev_S = self.S.get_arr()[pose]
        S_update = 0
        for x, y in np.ndindex(self.S.get_arr().shape):
            c = np.exp(np.log(inv_sensor_model / (1 - inv_sensor_model)) + np.log(prev_S / (1.00001 - prev_S)))
            S_update = c/(1 + c)
        self.S.update(S_update, pose)

    def update_dynamic_grid(self, inv_sensor_model, pose):
        prev_D = self.D.get_arr()[pose]
        D_update = 0
        for x, y in np.ndindex(self.D.get_arr().shape):
            c = np.exp(np.log(inv_sensor_model / (1 - inv_sensor_model)) + np.log(prev_D / (1.00001 - prev_D)))
            D_update = c / (1 + c)
        self.D.update(D_update, pose)


    def step(self,action):
        self.time += 1
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
