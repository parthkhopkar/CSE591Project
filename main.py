import numpy as np
import math
from robot import Robot


def get_observation(pose, env):
    dim = env.shape
    env = np.pad(env, [(1, 1), (1, 1)], constant_values=-1)
    x, y = pose
    r1 = x
    r2 = r1 + 3
    c1 = y
    c2 = y + 3
    # if x-1 < 0:
    #     c1 = 0
    # else:
    #     c1 = x-1
    # if x+2 > dim[0]:
    #     c2 = dim[0]
    # else:
    #     c2 = x+2
    #
    # if y-1 < 0:
    #     r1 = 0
    # else:
    #     r1 = y-1
    # if y+2 > dim[1]:
    #     r2 = dim[0]
    # else:
    #     r2 = y+2
    return env[r1:r2, c1:c2]

if __name__ == "__main__":
    dim = (4, 4)
    env = np.zeros(dim)  # Array for the environment
    S = np.full(dim, 0.5)  # Array for Static occupancy grid
    D = np.full(dim, 0.5)  # Array for dynamic occupancy grid
    # print('S:', S)
    # print('D:', D)
    env[0, dim[1]-1] = 1
    env[1, 1] = 1
    env[2, 1] = 1
    print(env)

    # Static map update
    inv_sensor_S = 0.9
    S = update_static_grid(S, inv_sensor_S)
    # print('Static grid after update')
    # print(S)

    # Dynamic map update
    inv_sensor_D = 0.1
    D = update_dynamic_grid(D, inv_sensor_D)
    # print('Dynamic grid after update')
    # print(D)

    print(observation((0, 3), env))



