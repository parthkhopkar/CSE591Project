import numpy as np
import math

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
    # print(env)

    # Static map update
    inv_sensor_S = 0.9
    S = update_static_grid(S, inv_sensor_S)
    print(S)

    # Dynamic map update
    inv_sensor_D = 0.9
    D = update_dynamic_grid(D, inv_sensor_D)
