import numpy as np
import math


def observation(pose, env):
    dim = env.shape
    x, y = pose
    r1, r2, c1, c2 = 0, 0, 0, 0
    if x-1 < 0:
        c1 = 0
    else:
        c1 = x-1
    if x+2 > dim[0]:
        c2 = dim[0]
    else:
        c2 = x+2

    if y-1 < 0:
        r1 = 0
    else:
        r1 = y-1
    if y+2 > dim[1]:
        r2 = dim[0]
    else:
        r2 = y+2

    return env[r1:r2, c1:c2]

def update_static_grid(S, inv_sensor_model):
    prev_S = S.copy()
    S_update = np.zeros(S.shape)
    for x, y in np.ndindex(S.shape):
        c = math.e ** math.log(inv_sensor_model / (1 - inv_sensor_model)) + math.log(prev_S[x, y] / (1 - prev_S[x, y]))
        S_update[x, y] = c/(1 + c)
    return S_update


def update_dynamic_grid(D, inv_sensor_model):
    prev_D = D.copy()
    D_update = np.zeros(S.shape)
    for x, y in np.ndindex(D.shape):
        c = math.e ** math.log(inv_sensor_model / (1 - inv_sensor_model)) + math.log(prev_D[x, y] / (1 - prev_D[x, y]))
        D_update[x, y] = c / (1 + c)
    return D_update


if __name__ == "__main__":
    dim = (4, 4)
    env = np.zeros(dim)  # Array for the environment
    S = np.full(dim, 0.5)  # Array for Static occupancy grid
    D = np.full(dim, 0.5)  # Array for dynamic occupancy grid
    # print('S:', S)
    # print('D:', D)
    env[0, dim[1]-1] = 1
    env[1, 1] = 1
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

    print(observation((0,1), env))

