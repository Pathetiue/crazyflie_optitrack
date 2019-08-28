
"""
Autonomous flight library for Crazyflie2
"""
import os
import sys
sys.path.append("../lib")
import time
from threading import Thread, Timer

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
from lqr import lqr


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
        self._cf = Crazyflie()
        self.t = Sensors.logs(self)
        # the three function calls below setup threads for connection monitoring
        self._cf.disconnected.add_callback(self._disconnected) #first monitor thread checking for disconnections
        self._cf.connection_failed.add_callback(self._connection_failed) #second monitor thread for checking for back connection to crazyflie
        self._cf.connection_lost.add_callback(self._connection_lost) # third monitor thread checking for lost connection
        print("Connecting to %s" % link_uri)
        self._cf.open_link(link_uri) #connects to crazyflie and downloads TOC/Params
        self.is_connected = True

        self.daemon = True
        self.timePrint = 0.0
        self.is_flying = False

        # Control Parm
        self.g = 10.  # gravitational acc. [m/s^ 2]
        self.m = 0.044  # mass [kg]
        self.pi = 3.14
        self.num_state = 6
        self.num_action = 3
        self.hover_thrust = 36850.0
        self.thrust2input = 115000
        self.input2thrust = 11e-6

        # trans calculated control to command input
        self.K_pitch = 1
        self.K_roll = 1
        # self.K_thrust = self.hover_thrust / (self.m * self.g)
        self.K_thrust = 4000

        # Logged states - ,
        # log.position, log.velocity and log.attitude are all in the body frame of reference
        self.position = [0.0, 0.0, 0.0]  # [m] in the global frame of reference
        self.velocity = [0.0, 0.0, 0.0]  # [m/s] in the global frame of reference
        self.attitude = [0.0, 0.0, 0.0]  # [rad] Attitude (p,r,y) with inverted roll (r)

        # References
        self.position_reference = [0.0, 0.0, 0.0]  # [m] in the global frame
        self.yaw_reference = 0.0  # [rad] in the global rame

        # Increments
        self.position_increments = [0.1, 0.1, 0.1]  # [m]
        self.micro_height_increments = 0.01
        self.yaw_increment = 0.1  # [rad]

        # Limits
        self.thrust_limit = (0, 63000)
        # self.roll_limit = (-30.0, 30.0)
        # self.pitch_limit = (-30.0, 30.0)
        # self.yaw_limit = (-200.0, 200.0)
        self.roll_limit = (-30, 30)
        self.pitch_limit = (-30, 30)

        # Controller settings
        self.isEnabled = True
        self.rate = 50 #Hz



    def _run_controller(self):
        """ Main control loop """
        # Controller parameters

        horizon = 20

        # Set the current reference to the current positional estimate, at a
        # slight elevation
        # time.sleep(2)
        self.position_reference = [0., -2., 0.5]



        # Unlock the controller, BE CAREFLUE!!!
        self._cf.commander.send_setpoint(0, 0, 0, 0)

        state_data = []
        reference_data = []
        save_dir = 'data' # will cover the old ones
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            os.makedirs(save_dir + '/training_data')
        step_count = 0
        while True:

            timeStart = time.time()

            # # tracking
            # x_r, y_r, z_r = [0., -2., 0.2]
            # dx_r, dy_r, dz_r = [0.0, 0.0, 0.0]

            x_r, y_r, z_r = self.position_reference
            dx_r, dy_r, dz_r = [0.0, 0.0, 0.0]

            # x_r, y_r, z_r = [0., 0., 1.]
            # dx_r, dy_r, dz_r = [0.0, 0.0, 0.5]

            target = np.array([x_r, y_r, z_r, dx_r, dy_r, dz_r])
            # print("position_references", self.position_reference)
            print("target: ", target)
            # Get measurements from the log
            x, y, z = self.position
            dx, dy, dz = self.velocity
            roll, pitch, yaw = self.t.attitude
            state = np.array([x, y, z, dx, dy, dz])
            print("state: ", state)

            state_data.append(state)
            reference_data.append(target)

            # Compute control signal - map errors to control signals
            if self.isEnabled:
                # mpc
                # mpc_policy = mpc(state, target, horizon)
                # roll_r, pitch_r, thrust_r = mpc_policy.solve()

                # lqr
                lqr_policy = lqr(state, target, horizon)
                roll_r, pitch_r, thrust_r = lqr_policy.solve()
                print("roll_computed: ", roll_r)
                print("pitch_computed: ", pitch_r)
                print("thrust_computed: ", thrust_r)

                roll_r = self.saturate(roll_r/self.pi*180, self.roll_limit)
                pitch_r = self.saturate(pitch_r/self.pi*180, self.pitch_limit)
                thrust_r = self.saturate((thrust_r + self.m * self.g) * self.thrust2input, self.thrust_limit)  # minus, see notebook
                # print("self.m * self.g", self.m * self.g)
                # print("input thrust_r: ", thrust_r)
            else:
                # If the controller is disabled, send a zero-thrust
                roll_r, pitch_r, thrust_r = (0, 0, 0)
            # Communicate a reference value to the Crazyflie
            # print("yaw angle: ", self.t.attitude[2])
            yaw_r = 0
            print("roll_r: ", roll_r)
            print("pitch_r: ", pitch_r)
            print("thrust_r: ", int(thrust_r))
            self._cf.commander.send_setpoint(roll_r, - pitch_r, yaw_r, int(thrust_r)) # change!!!

            # test height control
            # self._cf.commander.send_setpoint(0, 0, 0, int(thrust_r)) # change!!!

            step_count += 1
            if step_count > 500:
                np.save(save_dir + '/training_data/data' + str(step_count) + '.npy', state_data)
                np.save(save_dir + '/training_data/data' + str(step_count) + '.npy', reference_data)



            '''
            # Compute control errors
            ex = x_r - x
            ey = y_r - y
            ez = z_r - z
            dex = dx_r - dx
            dey = dy_r - dy
            dez = dz_r - dz

            Kzp, Krp, Kpp, Kyp = (200.0, 20.0, 20.0, 10.0)  # Pretty haphazard tuning
            Kzd, Krd, Kpd = (2 * 2 * sqrt(Kzp), 2 * sqrt(Krp), 2 * sqrt(Kpp))

            # Compute control signal - map errors to control signals
            if self.isEnabled:
                ux = +self.saturate(Krp * ex + Krd * dex, self.pitch_limit)
                uy = -self.saturate(Kpp * ey + Kpd * dey, self.roll_limit)
                pitch_r = cos(yaw) * ux - sin(yaw) * uy
                roll_r = sin(yaw) * ux + cos(yaw) * uy
                # thrust_r = + self.saturate((Kzp * ez + Kzd * dez + self.m * self.g) * (self.hover_thrust/(self.m*self.g)/ (cos(roll) * cos(pitch))), self.thrust_limit)
                thrust_r = + self.saturate((Kzp * ez + self.m * self.g) * (self.hover_thrust / (self.m * self.g) / (cos(roll) * cos(pitch))), self.thrust_limit)

            else:
                # If the controller is disabled, send a zero-thrust
                roll_r, pitch_r, yaw_r, thrust_r = (0, 0, 0, 0)

            # Communicate a reference value to the Crazyflie
            # self._cf.commander.send_setpoint(roll_r, pitch_r, 0, int(thrust_r))
            self._cf.commander.send_setpoint(0, 0, 0, int(thrust_r))
            '''
            self.loop_sleep(timeStart) # to make sure not faster than 50Hz

    def update_vals(self):
        self.position = self.t.position
        self.velocity = self.t.velocity  # [m/s] in the global frame of reference
        self.attitude = self.t.attitude  # [rad] Attitude (p,r,y) with inverted roll (r)

        Timer(.1, self.update_vals).start()

    def saturate(self, value, limits):
        """ Saturates a given value to reside on the the interval 'limits'"""
        if value < limits[0]:
            value = limits[0]
        elif value > limits[1]:
            value = limits[1]
        return value

    def print_at_period(self, period, message):
        """ Prints the message at a given period """
        if (time.time() - period) > self.timePrint:
            self.timePrint = time.time()
            print(message)

    def loop_sleep(self, timeStart):
        """ Sleeps the control loop to make it run at a specified rate """
        deltaTime = 1.0 / float(self.rate) - (time.time() - timeStart)
        if deltaTime > 0:
            print("50Hz")
            time.sleep(deltaTime)

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
            self.position_reference[2] += self.micro_height_increments
            if verbose: print('-z')
        if message == "m":
            self.position_reference[2] += self.micro_height_increments
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
        z-translation - controlled by ("i","k")
        yaw           - controlled by ("j","l")

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

    if len(available) > 0:
        le = Crazy_Auto(available[0][0])
        while le.is_connected:
            time.sleep(1)

    else:
        print("No Crazyflies found, cannot run example")