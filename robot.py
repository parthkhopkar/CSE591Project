import numpy as np
from occupancy_grid import OccupancyGrid
import math
import time as t

class Robot(object):

    def __init__(self, name=0, init_coord=(1, 1), dims=(20, 20), seed=100, dynamic_detection=True):
        self.seed = seed
        self.dynamic_detection = dynamic_detection
        self.time = 0
        self.pos = init_coord
        self.name = name
        self.limits = dims
        self.obs = None
        self.dim = dims
        self.obs_grid_size = 3
        self.gridSize = 5.0
        self.S = OccupancyGrid(False, dims[0], dims[1])
        self.D = OccupancyGrid(False, dims[0], dims[1])
        self.T = OccupancyGrid(length=dims[0], width=dims[1])
        self.dynamic_objects = []  # Store list of RFIDs of dynamic objectsS
        # self.env = OccupancyGrid(False, dims[0], dims[1])  # Occupancy Grid version of environment
        self.C = 80.0
        self.DT = 0.1  # time tick [s]
        self.SIM_TIME = 150.0  # simulation time [s]
        self.MAX_RANGE = int(( self.obs_grid_size * self.gridSize ) / math.sqrt(2) ) + 2.0 # maximum observation range
        self.M_DIST_TH = 2.0  # Threshold of Mahalanobis distance for data association.
        self.STATE_SIZE = 3  # State size [x,y,yaw]
        self.LM_SIZE = 2  # LM state size [x,y]
        # EKF state covariance
        self.Cx = np.diag([0.5, 0.5, np.deg2rad(30.0)]) ** 2

        #  Simulation parameter
        self.Q_sim = np.diag([0.2, np.deg2rad(1.0)]) ** 2
        self.R_sim = np.diag([1.0, np.deg2rad(10.0)]) ** 2
        # self.xEst = init_coord
        # self.PEst = np.eye(self.STATE_SIZE)
        # self.xDR = np.zeros((self.STATE_SIZE, 1))  # Dead reckoning

    # def get_estimate(self):
    #     return self.xEst, self.PEst, self.xDR

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
        self.obs = observation  # 3*3
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

    def ekf_slam(self, xEst, PEst, u, z):
        # Remove spurious landmark from xEst
        # remove_list = []
        # for i in range(3, len(xEst), 2):
        #     X, Y = int(xEst[i] / self.gridSize), int(xEst[i + 1] / self.gridSize)
        #     if self.S.get_arr()[X, Y] < 0.5:
        #         remove_list.append(i)
        #         remove_list.append(i + 1)
        # xEst = np.delete(xEst, remove_list)
        # xEst = xEst.reshape((xEst.shape[0], 1))
        # print(len(xEst))

        # Predict
        S = self.STATE_SIZE
        xEst[0:S] = self.motion_model(xEst[0:S], u)
        G, Fx = self.jacob_motion(xEst[0:S], u)
        PEst[0:S, 0:S] = G.T * PEst[0:S, 0:S] * G + Fx.T * self.Cx * Fx
        initP = np.eye(2)
        # Update
        #print('Observation')
        # print(z)
        for iz in range(len(z[:, 0])):  # for each observation
            if self.dynamic_detection:
                # If z does not correspond to a static landmark, continue
                x, y, theta = xEst[0], xEst[1], xEst[2]
                x_lm = x + z[iz, 0] * math.cos(theta + z[iz, 1])
                y_lm = y + z[iz, 0] * math.sin(theta + z[iz, 1])
                X, Y = int(x_lm/self.gridSize), int(y_lm/self.gridSize)
                # print(x_lm, X, y_lm, Y)
                # print('S', X, Y, self.S.get_arr()[X, Y])
                if self.S.get_arr()[X, Y] < 0.9:
                    # print(Robot, self.name, "not using %d, %d for localization\n"%(X, Y))
                    continue

            min_id = self.search_correspond_landmark_id(xEst, PEst, z[iz, 0:2])

            # if z[iz, 2] in self.dynamic_objects:
            #     print('Not using', z[iz, 2], 'for localization')
            #     continue

            nLM = self.calc_n_lm(xEst)

            if min_id == nLM:
                # print("New LM")
                # Extend state and covariance matrix
                xAug = np.vstack((xEst, self.calc_landmark_position(xEst, z[iz, :])))
                PAug = np.vstack((np.hstack((PEst, np.zeros((len(xEst), self.LM_SIZE)))),
                                  np.hstack((np.zeros((self.LM_SIZE, len(xEst))), initP))))
                xEst = xAug
                PEst = PAug
            lm = self.get_landmark_position_from_state(xEst, min_id)

            y, S, H = self.calc_innovation(lm, xEst, PEst, z[iz, 0:2], min_id)

            K = (PEst @ H.T) @ np.linalg.inv(S)
            xEst = xEst + (K @ y)
            PEst = (np.eye(len(xEst)) - (K @ H)) @ PEst

        xEst[2] = self.pi_2_pi(xEst[2])
        return xEst, PEst

    def calc_input(self, name, time, v=1.0, yaw_rate=0.1):
        # v = 1.0  # [m/s]
        # yaw_rate = 0.1  # [rad/s]
        if name == 1 and 10 < time < 20:
            u = np.array([[0, 0]]).T
        else:
            u = np.array([[v, yaw_rate]]).T
        return u

    def getDynamicObjects(self, D, z):
        if (time > 9):
            return [3]
        else:
            return []

    def observation(self, xTrue, xd, u, RFID, time):
        np.random.seed(self.seed)
        xTrue = self.motion_model(xTrue, u)

        # add noise to gps x-y
        z = np.zeros((0, 3))

        for i in range(len(RFID[:, 0])):

            dx = RFID[i, 0] - xTrue[0, 0]
            dy = RFID[i, 1] - xTrue[1, 0]
            d = math.sqrt(dx ** 2 + dy ** 2)
            angle = self.pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0])
            if d <= self.MAX_RANGE: #Make this consistent with the other obseravtion model
                dn = d + np.random.randn() * self.Q_sim[0, 0] ** 0.5  # add noise
                angle_n = angle + np.random.randn() * self.Q_sim[1, 1] ** 0.5  # add noise
                zi = np.array([dn, angle_n, i])
                # zi = np.array([d, angle, i])
                z = np.vstack((z, zi))

        # add noise to input
        ud = np.array([[
            u[0, 0] + np.random.randn() * self.R_sim[0, 0] ** 0.5,
            u[1, 0] + np.random.randn() * self.R_sim[1, 1] ** 0.5]]).T

        xd = self.motion_model(xd, ud)

        """
        Code added for updating occupancy grids
        """
        # Get OG(Occupancy Grid) position of robot
        x, y, theta = xTrue  # x, y = continuous location
        X, Y = int(x/self.gridSize), int(y/self.gridSize)
        obs_list = []
        env = np.zeros((self.dim[0], self.dim[1]))  # Occupancy Grid version of environment
        #print('Observation', z)
        for iz in range(len(z[:, 0])):
            x_obs = x + z[iz, 0] * math.cos(theta + z[iz, 1])
            y_obs = y + z[iz, 0] * math.sin(theta + z[iz, 1])
            #t.sleep(1)
            obs_list.append((x_obs, y_obs, z[iz, 2]))
            if self.D.get_arr()[int(x_obs/self.gridSize), int(y_obs/self.gridSize)] > 0.6 and iz not in self.dynamic_objects:
                self.dynamic_objects.append(iz)
            env[int(x_obs/self.gridSize), int(y_obs/self.gridSize)] = 1

        observation = self.get_observation((X, Y), env)

        list_global = {}
        # print(np.rot90(observation))
        for i in range(self.obs_grid_size):
            for j in range(self.obs_grid_size):
                if observation[i][j] == -1:
                    continue
                observation[i][j] = 0 if observation[i][j] >= 3 else observation[i][j]
                gx = X
                gy = Y
                # lx, ly = 3, 3
                shift_const = int(self.obs_grid_size/2)  # Assumes square observation
                nx = gx - (shift_const - i)
                ny = gy - (shift_const - j)
                list_global[(nx, ny)] = observation[i][j]

        for pose, obs in list_global.items():
            # if(pose==(6,1) and obs == 0):
            #     print(X,Y,obs_list)
            # print(pose, obs)
            self.update_static_grid(self.get_static_inv_sensor_model(pose, obs), pose)
            self.update_dynamic_grid(self.get_dynamic_inv_sensor_model(pose, obs), pose)
            self.T.update(time, pose)

        self.update_time(time)


        # dObj = self.getDynamicObjects(time)
        # Check if any observation is a dynamic object
        # for x, y, iz in obs_list:
        #     if self.D.get_arr()[int(x/self.gridSize), int(y/self.gridSize)] == 1:
        #         self.dynamic_objects.append(iz)

        # print(X, Y)
        # print(z)
        np.set_printoptions(linewidth=np.inf, precision=3, suppress=True)
        # print(np.rot90(self.T.get_arr()))
        #print('Static')
        #print(np.rot90(self.S.get_arr()))
        #print('Dynamic')
        #print(np.rot90(self.D.get_arr()))
        # print(np.rot90(env))
        return xTrue, z, xd, ud

    def get_observation(self, pose, env):
        pad = int(self.obs_grid_size/2)
        env1 = env.copy()
        env1 = np.pad(env1, [(pad, pad), (pad, pad)], mode='constant', constant_values=-1)
        x, y = pose
        r1 = x
        r2 = r1 + self.obs_grid_size
        c1 = y
        c2 = y + self.obs_grid_size
        return env1[r1:r2, c1:c2]

    def motion_model(self, x, u):
        F = np.array([[1.0, 0, 0],
                      [0, 1.0, 0],
                      [0, 0, 1.0]])

        B = np.array([[self.DT * math.cos(x[2, 0]), 0],
                      [self.DT * math.sin(x[2, 0]), 0],
                      [0.0, self.DT]])

        x = (F @ x) + (B @ u)
        return x

    def calc_n_lm(self, x):
        n = int((len(x) - self.STATE_SIZE) / self.LM_SIZE)
        return n

    def jacob_motion(self, x, u):
        Fx = np.hstack((np.eye(self.STATE_SIZE), np.zeros(
            (self.STATE_SIZE, self.LM_SIZE * self.calc_n_lm(x)))))

        jF = np.array([[0.0, 0.0, -self.DT * u[0] * math.sin(x[2, 0])],
                       [0.0, 0.0, self.DT * u[0] * math.cos(x[2, 0])],
                       [0.0, 0.0, 0.0]])

        G = np.eye(self.STATE_SIZE) + Fx.T * jF * Fx

        return G, Fx,

    def calc_landmark_position(self, x, z):
        zp = np.zeros((2, 1))

        zp[0, 0] = x[0, 0] + z[0] * math.cos(x[2, 0] + z[1])
        zp[1, 0] = x[1, 0] + z[0] * math.sin(x[2, 0] + z[1])

        return zp

    def get_landmark_position_from_state(self, x, ind):
        lm = x[self.STATE_SIZE + self.LM_SIZE * ind: self.STATE_SIZE + self.LM_SIZE * (ind + 1), :]

        return lm

    def search_correspond_landmark_id(self, xAug, PAug, zi):
        """
    Landmark association with Mahalanobis distance
    """

        nLM = self.calc_n_lm(xAug)

        min_dist = []

        for i in range(nLM):
            lm = self.get_landmark_position_from_state(xAug, i)
            y, S, H = self.calc_innovation(lm, xAug, PAug, zi, i)
            min_dist.append(y.T @ np.linalg.inv(S) @ y)

        min_dist.append(self.M_DIST_TH)  # new landmark

        min_id = min_dist.index(min(min_dist))

        return min_id

    def calc_innovation(self, lm, xEst, PEst, z, LMid):
        delta = lm - xEst[0:2]
        q = (delta.T @ delta)[0, 0]
        z_angle = math.atan2(delta[1, 0], delta[0, 0]) - xEst[2, 0]
        zp = np.array([[math.sqrt(q), self.pi_2_pi(z_angle)]])
        y = (z - zp).T
        y[1] = self.pi_2_pi(y[1])
        H = self.jacob_h(q, delta, xEst, LMid + 1)
        S = H @ PEst @ H.T + self.Cx[0:2, 0:2]

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

    def update_time(self, time):
        observed = np.array(self.T.get_arr() > 0, np.int)
        delT = np.full((self.limits[0], self.limits[1]), time) * observed - self.T.get_arr()
        delT = delT / self.C
        self.T.mat = self.T.mat * (1 - delT)

    def merge(self, robot_S, robot_D, robot_T):
        for i in range(self.limits[0]):
            for j in range(self.limits[1]):
                if robot_T.get_arr()[i, j] > self.T.get_arr()[i, j]:
                    self.S.update(robot_S.get_arr()[i, j], (i, j))
                    self.D.update(robot_D.get_arr()[i, j], (i, j))
                    self.T.update(robot_T.get_arr()[i, j], (i, j))

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
            c = np.exp(np.log(inv_sensor_model / (1.00001 - inv_sensor_model)) + np.log(prev_S / (1.00001 - prev_S)))
            S_update = c / (1 + c)
        self.S.update(S_update, pose)

    def update_dynamic_grid(self, inv_sensor_model, pose):
        prev_D = self.D.get_arr()[pose]
        D_update = 0
        for x, y in np.ndindex(self.D.get_arr().shape):
            c = np.exp(np.log(inv_sensor_model / (1.000001 - inv_sensor_model)) + np.log(prev_D / (1.00001 - prev_D)))
            D_update = c / (1 + c)
        self.D.update(D_update, pose)

    def step(self, action):
        self.time += 1
        if action == "r":
            return self.move_right()
        elif action == "o":
            return self.get_position()
        elif action == "l":
            return self.move_left()
        elif action == "u":
            return self.move_up()
        elif action == "d":
            return self.move_down()


if __name__ == "__main__":
    og = Robot()
    og.show()
