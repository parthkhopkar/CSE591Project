import numpy as np
import itertools
from robot import Robot
from occupancy_grid import OccupancyGrid
import json
import matplotlib.pyplot as plt


def get_observation(pose, env):
    env1 = env.get_arr().copy()
    env1 = np.pad(env1, [(1, 1), (1, 1)], mode='constant', constant_values=-1)
    x, y = pose
    r1 = x
    r2 = r1 + 3
    c1 = y
    c2 = y + 3
    return env1[r1:r2, c1:c2]

def runMapping(typ="MULTI_ROBOT"):
    if(typ == 'MULTI_ROBOT'):
        """
            2 robot test in 10X10 world
        """
        # Load world
        with open('./worlds/10X10.json') as file:
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
            for x in range(10):
                if x is 2:
                    world.step()
                for a0, a1 in itertools.zip_longest(maze['r0_actions'], maze['r1_actions']):
                    world.show()
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
                r0 = (robots[0].S.get_arr().copy(), robots[0].D.get_arr().copy(), robots[0].T.get_arr().copy())
                robots[0].merge(robots[1].S.get_arr(), robots[1].D.get_arr(), robots[1].T.get_arr())
                robots[1].merge(r0[0], r0[1], r0[2])

            world.show()
            np.set_printoptions(precision=1, suppress=True)
            print('Static Occupancy Grid for R0)')
            print(robots[0].S.get_arr())
            print('Dynamic Occupancy Grid for R0)')
            print(robots[0].D.get_arr())
            print('Time Map for R0)')
            print(robots[0].T.get_arr())
            print('Static Occupancy Grid for R1)')
            print(robots[1].S.get_arr())
            print('Dynamic Occupancy Grid for R1)')
            print(robots[1].D.get_arr())
            print('Time Map for R1)')
            print(robots[1].T.get_arr())
    else:
        """
        Single robot test
        """
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
        actions = ['r', 'r', 'u', 'u', 'u', 'l', 'l', 'd', 'd', 'd']
        for x in range(10):
            for action in actions:
                R1.setObs(get_observation(R1.step('o'), env))
                # print('Static Occupancy Grid')
                # print(R1.S.get_arr())
                # print('Dynamic Occupancy Grid')
                # print(R1.D.get_arr())
                old_pose = R1.get_position().copy()
                if old_pose[1] == 0 and old_pose[0] == 1 and x == 2:
                    env.step()
                    env.show()
                new_pose = R1.step(action)
                env.update_robot_position(R1.name, old_pose, new_pose)
        np.set_printoptions(precision=4)
        print('Static Occupancy Grid')
        print(np.round(R1.S.get_arr()))
        print('Dynamic Occupancy Grid')
        print(np.round(R1.D.get_arr()))


def runEKFSLAM():
    np.random.seed(42)

    # EKF state covariance
    Cx = np.diag([0.5, 0.5, np.deg2rad(30.0)]) ** 2

    #  Simulation parameter
    Q_sim = np.diag([0.2, np.deg2rad(1.0)]) ** 2
    R_sim = np.diag([1.0, np.deg2rad(10.0)]) ** 2

    DT = 0.1  # time tick [s]
    SIM_TIME = 150.0  # simulation time [s]
    MAX_RANGE = 20.0  # maximum observation range
    M_DIST_TH = 2.0  # Threshold of Mahalanobis distance for data association.
    STATE_SIZE = 3  # State size [x,y,yaw]
    LM_SIZE = 2  # LM state size [x,y]

    show_animation = True

    print(__file__ + " start!!")

    time = 0.0

    # RFID positions [x, y]
    RFID = np.array([[7.0, 23.0],
                     [23.0, 13.0],
                     [7.0, 13.0],
                     [23.0, 63.0],
                     [27.0, 17.5],
                     [17.0, 27.0],
                     [17.0, 3.0],
                     [32.5, 7.0]])

    # State Vector [x y yaw v]'
    #xEst = np.zeros((STATE_SIZE, 1))
    xTrue = np.array([[20.0],[5.0],[0.0]])
    xEst = np.array([[20.0],[5.0],[0.0]])
    PEst = np.eye(STATE_SIZE)

    xDR = np.zeros((STATE_SIZE, 1))  # Dead reckoning

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    ts = 0

    robot = Robot()
    while SIM_TIME >= time:
        print(robot.dynamic_objects)
        ts += 1
        time += DT
        # print(time)
        if time > 40:
            #print('LM changed')
            RFID = np.array([[7.0, 23.0],
                     [23.0, 13.0],
                     [7.0, 13.0],
                     [23.0, 63.0],
                     [27.0, 34.0],
                     [17.0, 27.0],
                     [17.0, 3.0],
                     [32.5, 7.0]])
        if time > 70:
            #print('LM changed')
            RFID = np.array([[7.0, 23.0],
                     [23.0, 13.0],
                     [7.0, 13.0],
                     [23.0, 63.0],
                     [27.0, 34.0],
                     [12.0, 17.0],
                     [17.0, 3.0],
                     [32.5, 7.0]])

        u = robot.calc_input()

        xTrue, z, xDR, ud = robot.observation(xTrue, xDR, u, RFID, time)
                
        xEst, PEst = robot.ekf_slam(xEst, PEst, ud, z)
        
        #print(len(xEst))
            

        x_state = xEst[0:STATE_SIZE]

        # store data history
        hxEst = np.hstack((hxEst, x_state))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        
        errorX = hxTrue[0, :] - hxEst[0, :]
        errorY = hxTrue[1, :] - hxEst[1, :]
        
        error = np.sum(np.sqrt(errorX**2 + errorY**2)) / ts
        
        #print(error)

        if show_animation:  # pragma: no cover
            plt.cla()

            plt.plot(RFID[:, 0], RFID[:, 1], "*k")
            plt.plot(xEst[0], xEst[1], ".r")

            # plot landmark
            for i in range(robot.calc_n_lm(xEst)):
                plt.plot(xEst[STATE_SIZE + i * 2],
                         xEst[STATE_SIZE + i * 2 + 1], "xg")

            plt.plot(hxTrue[0, :],
                     hxTrue[1, :], "-b")
#             plt.plot(hxDR[0, :],
#                      hxDR[1, :], "-k")
            plt.plot(hxEst[0, :],
                     hxEst[1, :], "-r")
            
            time_text = plt.text(10.02, 10.95, '')
            
            time_text2 = plt.text(45, 50.95, '')
            
            
            time_text.set_text('err = %.1f' % error)
            time_text2.set_text('time = %.1f' % time)
            
            plt.axis([0,50,0,50])
            major_ticks = np.arange(0, 50, 5)
            minor_ticks = np.arange(0, 50, 5)
            plt.xticks(major_ticks)
            plt.yticks(minor_ticks)
            plt.grid(True)
            plt.pause(0.001)
if __name__ == "__main__":
    runEKFSLAM()

