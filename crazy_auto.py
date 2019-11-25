
"""
Autonomous flight library for Crazyflie2
"""
import os
import sys
sys.path.append("../lib")
import time
from threading import Thread, Timer
import threading

import termios
import contextlib
import numpy as np
import logging
logging.basicConfig(level=logging.ERROR)

import Sensors

from cflib.crazyflie import Crazyflie
import cflib.crtp
from math import sin, cos, sqrt

from mpc import mpc
# from lqr import lqr


@contextlib.contextmanager
def raw_mode(file):
    """ Implement termios module for keyboard detection """
    old_attrs = termios.tcgetattr(file.fileno())
    new_attrs = old_attrs[:]
    new_attrs[3] = new_attrs[3] & ~(termios.ECHO | termios.ICANON)
    try:
        termios.tcsetattr(file.fileno(), termios.TCSADRAIN, new_attrs)
        yield
    finally:
        termios.tcsetattr(file.fileno(), termios.TCSADRAIN, old_attrs)


class Crazy_Auto:
    """ Basic calls and functions to enable autonomous flight """  
    def __init__(self, link_uri):
        """ Initialize crazyflie using passed in link"""
        self.s1 = threading.Semaphore(1)
        self._cf = Crazyflie()
        self.t = Sensors.logs(self)

        # the three function calls below setup threads for connection monitoring
        self._cf.disconnected.add_callback(self._disconnected) #first monitor thread checking for disconnections
        self._cf.connection_failed.add_callback(self._connection_failed) #second monitor thread for checking for back connection to crazyflie
        self._cf.connection_lost.add_callback(self._connection_lost) # third monitor thread checking for lost connection
        print("Connecting to %s" % link_uri)
        self._cf.open_link(link_uri) #connects to crazyflie and downloads TOC/Params
        self.is_connected = True

        # Control Parm
        self.g = 10.  # gravitational acc. [m/s^ 2]
        self.m = 0.044  # mass [kg]
        self.pi = 3.14
        self.num_state = 6
        self.num_action = 3
        self.hover_thrust = 36850.0
        self.thrust2input = 115000

        # Logged states - ,
        # log.position, log.velocity and log.attitude are all in the body frame of reference
        self.position = [0.0, 0.0, 0.0]  # [m] in the global frame of reference
        self.velocity = [0.0, 0.0, 0.0]  # [m/s] in the global frame of reference
        self.attitude = [0.0, 0.0, 0.0]  # [rad] Attitude (p,r,y) with inverted roll (r)

        # References
        self.position_reference = [0.00, 0.00, 0.00]  # [m] in the global frame

        # Increments
        self.position_increments = [0.1, 0.1, 0.1, 0.02]  # [m]

        # Limits
        self.thrust_limit = (0, 63000)
        self.roll_limit = (-30, 30)
        self.pitch_limit = (-30, 30)

        # Controller settings
        self.isEnabled = True
        self.rate = 50 # Hz
        self.period = 1.0 / float(self.rate)


    def _run_controller(self):
        """ Main control loop """
        # Wait for feedback
        time.sleep(2)
        # Unlock the controller, BE CAREFLUE!!!
        self._cf.commander.send_setpoint(0, 0, 0, 0)

        horizon = 20
        step_count = 0

        self.position_reference = [0., 0., 0.7]

        # state_data = []
        # reference_data = []
        control_data = []
        save_dir = 'data' # will cover the old ones
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            os.makedirs(save_dir + '/training_data')


        self.last_time = time.time()
        while True:

            x_r, y_r, z_r = self.position_reference
            dx_r, dy_r, dz_r = [0.0, 0.0, 0.0]
            target = np.array([x_r, y_r, z_r, dx_r, dy_r, dz_r])

            x, y, z = self.position
            dx, dy, dz = self.velocity
            roll, pitch, yaw = self.attitude
            state = np.array([x, y, z, dx, dy, dz, roll, pitch, yaw])

            print("target: ", target)
            print("state: ", state)

            # state_data.append(state)
            # reference_data.append(target)

            # Compute control signal - map errors to control signals
            if self.isEnabled:
                if dx == 0.0 and dy == 0.0 and dz == 0.0:
                    roll_r = 0
                    pitch_r = 0
                    thrust_r = self.hover_thrust
                    print("NO FEEDBACK!")
                else:

                    # mpc
                    mpc_policy = mpc(state[:6], target, horizon)
                    roll_r, pitch_r, thrust_r = mpc_policy.solve()

                    # lqr
                    # lqr_policy = lqr(state, target, horizon)
                    # roll_r, pitch_r, thrust_r = lqr_policy.solve()
                    # print("roll_computed: ", roll_r)
                    # print("pitch_computed: ", pitch_r)
                    # print("thrust_computed: ", thrust_r)

                    roll_r = self.saturate(roll_r/self.pi*180, self.roll_limit)
                    pitch_r = self.saturate(pitch_r/self.pi*180, self.pitch_limit)
                    thrust_r = self.saturate((thrust_r + self.m * self.g) * self.thrust2input, self.thrust_limit)  # minus, see notebook
                    # thrust_r = self.saturate(
                    #     (thrust_r + self.m * self.g) / (cos(pitch/180.*self.pi) * cos(roll/180.*self.pi)) * self.thrust2input,
                    #     self.thrust_limit)  # minus, see notebook

            else:
                # If the controller is disabled, send a zero-thrust
                roll_r, pitch_r, thrust_r = (0, 0, 0)

            yaw_r = 0
            print("roll_r: ", roll_r)
            print("pitch_r: ", pitch_r)
            print("thrust_r: ", int(thrust_r))

            # control_data.append(np.concatenate([target, state, np.array([roll_r, - pitch_r, yaw_r, int(thrust_r), time.time()])]))
            control_data.append(time.time())
            step_count += 1
            if step_count % 5000 == 0:
                # np.save(save_dir + '/training_data/state' + str(step_count) + '.npy', state_data)
                # np.save(save_dir + '/training_data/ref' + str(step_count) + '.npy', reference_data)
                np.save(save_dir + '/training_data/control' + str(step_count) + '.npy', control_data)

            self.loop_sleep()  # to make sure not faster than 200Hz
            self._cf.commander.send_setpoint(roll_r, - pitch_r, yaw_r, int(thrust_r)) # change!!!
            # height = 30
            # self._cf.commander.send_hover_setpoint(0, 0, 0, height / 100.)

            '''
            ## PID
            # Compute control errors
            ex = x - x_r
            ey = y - y_r
            ez = z - z_r
            dex = dx - dx_r
            dey = dy - dy_r
            dez = dz - dz_r

            xi = 1.2
            wn = 3.0q
            Kp = - wn * wn
            Kd = - 2 * wn * xi

            Kxp = 1.2 * Kp
            Kxd = 1.2 * Kd
            Kyp = Kp
            Kyd = Kd
            Kzp = 0.8 * Kp
            Kzd = 0.8 * Kd
            # Compute control signal - map errors to control signals
            if self.isEnabled:
                ux = self.saturate(Kxp * ex + Kxd * dex, self.roll_limit)
                uy = self.saturate(Kyp * ey + Kyd * dey, self.pitch_limit)
                pitch_r = uy
                roll_r = ux
                # pitch_r = cos(yaw) * ux - sin(yaw) * uy
                # roll_r = sin(yaw) * ux + cos(yaw) * uy
                thrust_r = self.saturate((Kzp * ez + Kzd * dez + self.g) * self.m * self.thrust2input,
                                         self.thrust_limit)  # / (cos(roll) * cos(pitch))

            else:
                # If the controller is disabled, send a zero-thrust
                roll_r, pitch_r, yaw_r, thrust_r = (0, 0, 0, 0)
            yaw_r = 0
            # self._cf.commander.send_setpoint(roll_r, pitch_r, 0, int(thrust_r))
            # self._cf.commander.send_setpoint(0, 0, 0, int(thrust_r))
            print("Kp: ", Kp)
            print("Kd: ", Kd)
            print("z control: ", (((Kzp * ez + Kzd * dez + self.g) * self.m)))
            print("roll_r: ", roll_r)
            print("pitch_r: ", pitch_r)
            print("thrust_r: ", int(thrust_r))
            control_data.append(np.array([roll_r, pitch_r, yaw_r, int(thrust_r)]))
            '''

    def update_vals(self):
        self.s1.acquire()
        self.position = self.t.position
        self.velocity = self.t.velocity  # [m/s] in the global frame of reference
        self.attitude = self.t.attitude  # [rad] Attitude (p,r,y) with inverted roll (r)
        self.s1.release()
        # print("update_vals")
        Timer(.005, self.update_vals).start()

    def saturate(self, value, limits):
        """ Saturates a given value to reside on the the interval 'limits'"""
        if value < limits[0]:
            value = limits[0]
        elif value > limits[1]:
            value = limits[1]
        return value

    def loop_sleep(self):
        """ Sleeps the control loop to make it run at a specified rate """
        deltaTime = self.period - (time.time() - self.last_time)
        if deltaTime > 0:
            print("real_time")
            time.sleep(deltaTime)
        self.last_time = time.time()


    def set_reference(self, message):
        """ Enables an incremental change in the reference and defines the
        keyboard mapping (change to your preference, but if so, also make sure
        to change the valid_keys attribute in the interface thread)"""
        verbose = True
        if message == "s":
            self.position_reference[1] -= self.position_increments[1]
            if verbose: print('-y')
        if message == "w":
            self.position_reference[1] += self.position_increments[1]
            if verbose: print('+y')
        if message == "d":
            self.position_reference[0] += self.position_increments[1]
            if verbose: print('+x')
        if message == "a":
            self.position_reference[0] -= self.position_increments[1]
            if verbose: print('-x')
        if message == "k":
            self.position_reference[2] -= self.position_increments[2]
            if verbose: print('-z')
        if message == "i":
            self.position_reference[2] += self.position_increments[2]
            if verbose: print('+z')
        if message == "n":
            self.position_reference[2] += self.position_increments[3]
            if verbose: print('-z')
        if message == "m":
            self.position_reference[2] -= self.position_increments[3]
            if verbose: print('+z')
        # if message == "j":
        #     self.yaw_reference += self.yaw_increment
        # if message == "l":
        #     self.yaw_reference -= self.yaw_increment
        if message == "q":
            self.isEnabled = False
        if message == "e":
            self.isEnabled = True


### ------------------------------------------------------- Callbacks ----------------------------------------------------------------------###
    def _connected(self, link_uri):
        """ This callback is called form the Crazyflie API when a Crazyflie
        has been connected and the TOCs have been downloaded."""

        # Start a separate thread to do the motor test.
        # Do not hijack the calling thread!
        Thread(target=self.update_vals).start()
        print("Waiting for logs to initalize...")
        Thread(target=self._run_controller).start()
        master = inputThread(self)

    def _connection_failed(self, link_uri, msg):
        """Callback when connection initial connection fails (i.e no Crazyflie
        at the speficied address)"""
        print("Connection to %s failed: %s" % (link_uri, msg))
        self.is_connected = False

    def _connection_lost(self, link_uri, msg):
        """Callback when disconnected after a connection has been made (i.e
        Crazyflie moves out of range)"""
        print("Connection to %s lost: %s" % (link_uri, msg))
        self.is_connected = False

    def _disconnected(self, link_uri):
        """Callback when the Crazyflie is disconnected (called in all cases)"""

        print("Disconnected from %s" % link_uri)
        self.is_connected = False
### ------------------------------------------------------ END CALLBACKS -------------------------------------------------------------------###



class inputThread(Thread):
    """Create an input thread which sending references taken in increments from
    the keys "valid_characters" attribute. With the incremental directions (+,-)
    a mapping is done such that

        x-translation - controlled by ("w","s")
        y-translation - controlled by ("a","d")
        z-translation - controlled by ("i","k") ("m","n")

    Furthermore, the controller can be enabled and disabled by

        disable - controlled by "q"
        enable  - controlled by "e"
    """
    def __init__(self, controller):
        Thread.__init__(self)
        self.valid_characters = ["a","d","s", "w","i","j","l","k","q","e"]
        self.daemon = True
        self.controller = controller
        self.start()

    def run(self):
        with raw_mode(sys.stdin):
            try:
                while True:
                    ch = sys.stdin.read(1)
                    if ch in self.valid_characters:
                        self.controller.set_reference(ch)

            except (KeyboardInterrupt, EOFError):
                sys.exit(0)
                pass


if __name__ == '__main__':
    # Initialize the low-level drivers (don't list the debug drivers)
    cflib.crtp.init_drivers(enable_debug_driver=False)
    # Scan for Crazyflies and use the first one found
    print("Scanning interfaces for Crazyflies...")
    available = cflib.crtp.scan_interfaces()
    print("Crazyflies found:")
    for i in available:
        print(i[0])

    # le = Crazy_Auto('radio://0/80/2M/E7E7E7E702')
    # while le.is_connected:
    #     time.sleep(1)

    if len(available) > 0:
        le = Crazy_Auto(available[0][0])
        while le.is_connected:
            time.sleep(1)

    else:
        print("No Crazyflies found, cannot run example")
