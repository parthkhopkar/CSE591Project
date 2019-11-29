import json
import numpy as np
from robot import Robot
from occupancy_grid import OccupancyGrid

with open('.\\worlds\\10X10.json') as file:
    maze = json.load(file)
    # Initialize maze
    dim = [maze['dim1'], maze['dim2']]
    world = OccupancyGrid(True, dim[0], dim[1])
    # Initialize robots
    robots = []
    for i in range(maze['n']):
        name = i+3
        robots.append(Robot(name, maze['r'+str(i)], dim))
        world.update_robot_position(name, maze['r'+str(i)], maze['r'+str(i)])
    # Initialize static objects
    for i in maze['static']:
        world.add_static_object(i[0], i[1])

    world.show()