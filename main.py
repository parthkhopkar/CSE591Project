import numpy as np
import math
from robot import Robot
from occupancy_grid import OccupancyGrid

def get_observation(pose, env):
    env1 = env.get_arr().copy()
    env1 = np.pad(env1, [(1, 1), (1, 1)], constant_values=-1)
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
    return env1[r1:r2, c1:c2]

if __name__ == "__main__":
    dim = [4, 4]
    env = OccupancyGrid()  # Array for the environment
    env.add_static_object(2, 1)
    env.add_static_object(3, 3)
    env.add_dynamic_object(2, 3, 1, 3)
    start_pose = [3, 0]
    name = 3
    R1 = Robot(name, start_pose, dim)
    env.update_robot_position(name, start_pose, start_pose)
    # env.show()
    print(R1.S.get_arr())
    actions = ['r', 'r', 'u', 'u', 'u', 'l', 'l']
    for action in actions:
        R1.setObs(get_observation(R1.step('o'), env))
        print(R1.S.get_arr())
        env.show()
		env.step()
        old_pose = R1.get_position().copy()
        new_pose = R1.step(action)
        env.update_robot_position(R1.name, old_pose, new_pose)

    # # print('S:', S)
    # # print('D:', D)
    # env[0, dim[1]-1] = 1
    # env[1, 1] = 1
    # env[2, 1] = 1
    # print(env)
    #
    # # Static map update
    # inv_sensor_S = 0.9
    # S = update_static_grid(S, inv_sensor_S)
    # # print('Static grid after update')
    # # print(S)
    #
    # # Dynamic map update
    # inv_sensor_D = 0.1
    # D = update_dynamic_grid(D, inv_sensor_D)
    # # print('Dynamic grid after update')
    # # print(D)
    #
    # print(observation((0, 3), env))



