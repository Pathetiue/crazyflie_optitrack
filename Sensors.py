'''
Sensors
'''

import time
import threading
from cfclient.utils.logconfigreader import LogConfig
import logging
logging.basicConfig(level=logging.ERROR)
from NatNetClient import NatNetClient

class logs:
    def __init__(self, cf):
        #local copy of crazy_Auto
        self._cf = cf

        # Roll, Pitch, Yaw
        self.attitude = [0,0,0]
        # X, Y, Z
        self.position = [0,0,0]
        # Vx, Vy, Vz
        self.velocity = [0,0,0]

        self._cf._cf.connected.add_callback(self._init_flight_var)
        
    def _init_flight_var(self, link_uri):

        print("Connected to %s" % link_uri)

        self.RPY_log = LogConfig(name="Stabilizer", period_in_ms=10)
        self.RPY_log.add_variable("stabilizer.roll", "float")
        self.RPY_log.add_variable("stabilizer.pitch", "float")
        self.RPY_log.add_variable("stabilizer.yaw", "int16_t")
        self.RPY_log.add_variable('stabilizer.thrust', 'float')

        self._cf._cf.log.add_config(self.RPY_log)

        self.RPY_log.data_received_cb.add_callback(self.update_attitude)
        self.RPY_log.error_cb.add_callback(self.update_error)

        self.RPY_log.start()
        # self.quaternion.start()
        print("Logging Started\n")
        
        self._cf._cf.connected.add_callback(self._cf._connected)

        # optitrack stuff
        self.l_odom = list()
        self.l_index = -1
        self.sampleInterval = 10
        self.s1 = threading.Semaphore(1)

        self.streamingClient = NatNetClient("192.168.1.113")  # Net2
        self.streamingClient.rigidBodyListener = self.receiveRigidBodyFrame
        self.streamingClient.run()

    def update_error(self, logconf, msg):
        print("Error when logging %s: %s" % (logconf.name, msg))

    def update_attitude(self, timestamp, data, logconf):
        # print(data)
        self.attitude[0] = data["stabilizer.roll"]
        self.attitude[1] = data["stabilizer.pitch"]
        self.attitude[2] = data["stabilizer.yaw"]

    def receiveRigidBodyFrame(self, id, pos, rotation):
        msg = {
            'position': [0., 0., 0.],
            'stamp': 0,
            'velocity': [0., 0., 0.]
        }
        msg['stamp'] = time.time()
        msg['position'][0] = pos[0]
        msg['position'][1] = pos[1]
        msg['position'][2] = pos[2]
        self.s1.acquire()
        if len(self.l_odom) == self.sampleInterval:
            last_index = (self.l_index + 2) % self.sampleInterval
            # last_index = (self.l_index + 8) % self.sampleInterval
            last_msg = self.l_odom[last_index]
            deltatime = msg['stamp'] - last_msg['stamp']
            print("delta_time: ", deltatime)
            msg['velocity'][0] = (pos[0] - last_msg['position'][0]) / deltatime
            msg['velocity'][1] = (pos[1] - last_msg['position'][1]) / deltatime
            msg['velocity'][2] = (pos[2] - last_msg['position'][2]) / deltatime
            if abs(msg['velocity'][0]) < 0.0001:
                msg['velocity'][0] = 0
            if abs(msg['velocity'][1]) < 0.0001:
                msg['velocity'][1] = 0
        else:
            self.l_odom.append(msg)
        self.l_index = (self.l_index + 1) % self.sampleInterval
        self.l_odom[self.l_index] = msg
        self.s1.release()

        self.position[0] = msg['position'][0]
        self.position[1] = msg['position'][1]
        self.position[2] = msg['position'][2]

        self.velocity[0] = msg['velocity'][0]
        self.velocity[1] = msg['velocity'][1]
        self.velocity[2] = msg['velocity'][2]
        print("position: ", self.position)
        print("velocity: ", self.velocity)

    # for debug
    def log_file_print(self, file, data):
        for i in range(len(data)):
            file.write(str(data[i]))
            file.write(',')
        file.write('\n')

    def get_altitude(self):
        return self.position[2]

    def get_roll(self):
        print("roll:  ", self.attitude[0])

    def get_pitch(self):
        print("pitch:  ", self.attitude[1])

    def get_yaw(self):
        print("yaw:  ", self.attitude[2])


