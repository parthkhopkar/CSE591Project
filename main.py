import numpy as np
import itertools
from robot import Robot
from occupancy_grid import OccupancyGrid
import json


def get_observation(pose, env):
    env1 = env.get_arr().copy()
    env1 = np.pad(env1, [(1, 1), (1, 1)], constant_values=-1)
    x, y = pose
    r1 = x
    r2 = r1 + 3
    c1 = y
    c2 = y + 3
    return env1[r1:r2, c1:c2]

if __name__ == "__main__":
    """
        2 robot test in 10X10 world
    """
    # Load world
    with open('.\\worlds\\10X10.json') as file:
        maze = json.load(file)
        # Initialize maze
        dim = [maze['dim1'], maze['dim2']]
        world = OccupancyGrid(True, dim[0], dim[1])
        # Initialize robots
        robots = []
        for i in range(maze['n']):
            name = i + 3
            robots.append(Robot(name, maze['r' + str(i)], dim))
            world.update_robot_position(name, maze['r' + str(i)], maze['r' + str(i)])
        # Initialize static objects
        for obj in maze['static']:
            world.add_static_object(obj[0], obj[1])
        # Initialize Dynamic Objects
        for obj in maze['dynamic']:
            print(obj)
            world.add_dynamic_object(obj[0][0], obj[0][1], obj[1][0], obj[1][1])
        # Update poses
        for x in range(5):
            if x is 2:
                world.step()
            for a0, a1 in itertools.zip_longest(maze['r0_actions'], maze['r1_actions']):
                if a0:
                    robots[0].setObs(get_observation(robots[0].step('o'), world))
                    old_pose = robots[0].get_position().copy()
                    new_pose = robots[0].step(a0)
                    world.update_robot_position(robots[0].name, old_pose, new_pose)
                if a1:
                    robots[1].setObs(get_observation(robots[1].step('o'), world))
                    old_pose = robots[1].get_position().copy()
                    new_pose = robots[1].step(a1)
                    world.update_robot_position(robots[1].name, old_pose, new_pose)

        world.show()
        np.set_printoptions(precision=1, suppress=True)
        print('Static Occupancy Grid for R0)')
        print(robots[0].S.get_arr())
        print('Dynamic Occupancy Grid for R0)')
        print(robots[0].D.get_arr())
        print('Static Occupancy Grid for R1)')
        print(robots[1].S.get_arr())
        print('Dynamic Occupancy Grid for R1)')
        print(robots[1].D.get_arr())

    """
    Single robot test
    """
    # dim = [4, 4]
    # env = OccupancyGrid()  # Array for the environment
    # env.add_static_object(2, 1)
    # env.add_static_object(3, 3)
    # env.add_dynamic_object(2, 3, 1, 3)
    # start_pose = [3, 0]
    # name = 3
    # R1 = Robot(name, start_pose, dim)
    # env.update_robot_position(name, start_pose, start_pose)
    # # env.show()
    # actions = ['r', 'r', 'u', 'u', 'u', 'l', 'l', 'd', 'd', 'd']
    # for x in range(10):
    #     for action in actions:
    #         R1.setObs(get_observation(R1.step('o'), env))
    #         # print('Static Occupancy Grid')
    #         # print(R1.S.get_arr())
    #         # print('Dynamic Occupancy Grid')
    #         # print(R1.D.get_arr())
    #         old_pose = R1.get_position().copy()
    #         if old_pose[1] == 0 and old_pose[0] == 1 and x == 2:
    #             env.step()
    #             env.show()
    #         new_pose = R1.step(action)
    #         env.update_robot_position(R1.name, old_pose, new_pose)
    # np.set_printoptions(precision=4)
    # print('Static Occupancy Grid')
    # print(np.round(R1.S.get_arr()))
    # print('Dynamic Occupancy Grid')
    # print(np.round(R1.D.get_arr()))



