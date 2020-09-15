import sys, os
from configparser import ConfigParser
base_path = os.getcwd().split('src')[0]
sys.path.insert(1, base_path)
from src.utils import vrep
from src.utils.im_utils import *

config = ConfigParser()
config.read('config.cfg')


def get_coordinates(path, use_z=True, plot_fig=True):
    # use only X, Y 
    if use_z:
        coordinates = np.genfromtxt(path, delimiter=',', skip_header=2, usecols=(1, 2, 3), dtype=np.float)
    else:
        coordinates = np.genfromtxt(path, delimiter=',', skip_header=2, usecols=(1, 2), dtype=np.float)

    if plot_fig:
        plt.figure()
        plt.plot(coordinates[:, 0], coordinates[:, 1])
        plt.show()

    coordinates[:, 0] = (coordinates[:, 0] - coordinates[0, 0]) / 2.0
    coordinates[:, 1] = (coordinates[:, 1] - coordinates[0, 1]) / 2.0
    return coordinates


def draw(clientID, coordinates, final_xy, object_name, final_pos=False, use_z=True):
    res, objs = vrep.simxGetObjects(clientID, vrep.sim_handle_all, vrep.simx_opmode_oneshot_wait)
    if res == vrep.simx_return_ok:
        res, v0 = vrep.simxGetObjectHandle(clientID, object_name, vrep.simx_opmode_oneshot_wait)

        # Reads the pen position X,Y,Z simxGetVisionSensorImage
        res, pos = vrep.simxGetObjectPosition(clientID, v0, vrep.sim_handle_parent, vrep.simx_opmode_oneshot_wait)
        print("Initial Position", pos)
        # sys.exit()
        i = 0
        for coordinate in coordinates:
            time.sleep(0.05)

            # Sum X and Y
            if use_z:
                cmd_pos = np.array(pos) + coordinate
            else:
                cmd_pos = np.array(pos) + np.append(coordinate, [0], axis=0)
            print("Position: ", i, coordinate, cmd_pos)

            i += 1
            # Sets the new position
            res = vrep.simxSetObjectPosition(clientID, v0, vrep.sim_handle_parent, cmd_pos,
                                             vrep.simx_opmode_oneshot_wait)

            if res != 0:
                vrep.simxFinish(clientID)
                print('Remote API function call returned with error code: ', res)
                break

        # lift the pen by 0.05
        if use_z:
            cmd_pos = np.array(pos) + coordinates[-1]
        else:
            cmd_pos = np.array(pos) + np.append(coordinates[-1], [0.05], axis=0)
        res = vrep.simxSetObjectPosition(clientID, v0, vrep.sim_handle_parent, cmd_pos, vrep.simx_opmode_oneshot_wait)

        time.sleep(0.05)
        if final_pos:
            final_pos = np.append([final_xy], [cmd_pos[2]], axis=0)
            dif_pos = final_pos - cmd_pos
            dif_pos = dif_pos / 10.0

            for i in range(10):
                # lift the pen
                cmd_pos = cmd_pos + dif_pos
                res = vrep.simxSetObjectPosition(clientID, v0, vrep.sim_handle_parent, cmd_pos,
                                                 vrep.simx_opmode_oneshot_wait)
                time.sleep(0.05)
    else:
        print('Remote API function call returned with error code: ', res)


if __name__ == '__main__':
    # close all open connections
    vrep.simxFinish(-1)
    coordinates = get_coordinates(path=config['generate_motion']['final_motion'])
    final_xy = []
    vision_sensor = False
    object_name = 'feltPen_invisible'

    print('Simulation starts..')
    clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # 19999
    if clientID != -1:
        print('Connected to remote API server')
        if not vision_sensor:
            draw(clientID, coordinates, final_xy, object_name)
        else:
            try:
                stream_vision_sensor('Baxter_camera', clientID, 0.0001)
            except:
                draw(clientID, coordinates, final_xy, object_name)
                stream_vision_sensor('Baxter_camera', clientID, 0.0001)
        vrep.simxFinish(clientID)
    else:
        print('Connection non successful')
        sys.exit('Could not connect')
print('Simulation completed')
