
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
from lqr import FiniteHorizonLQR
from math import sin, cos, sqrt


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
        self.yaw_increment = 0.1  # [rad]

        # Limits
        self.thrust_limit = (0, 65000)
        # self.roll_limit = (-30.0, 30.0)
        # self.pitch_limit = (-30.0, 30.0)
        # self.yaw_limit = (-200.0, 200.0)
        self.roll_limit = (-3.0, 3.0)
        self.pitch_limit = (-3.0, 3.0)

        # Controller settings
        self.isEnabled = True
        self.rate = 50 #Hz



        # Control Parm
        self.g = 9.8  # gravitational acc. [m/s^ 2]
        self.m = 0.027  # mass [kg]
        self.pi = 3.14
        self.num_state = 6
        self.num_action = 3
        self.hover_thrust = 36850.0

        # trans calculated control to command input
        self.K_pitch = 1
        self.K_roll = 1
        # self.K_thrust = self.hover_thrust / (self.m * self.g)
        self.K_thrust = 4000

        self.A = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])

        self.B = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [self.g, 0, 0],
            [0, -self.g, 0],
            [0, 0, 1.0/0.044]
        ]) # assume the yaw angle is 0

        Qdiag = np.zeros(self.num_state)
        Qdiag[0:2] = 10.  # position
        Qdiag[2] = 5.
        Qdiag[3:6] = 1. # velosity
        self.Q = np.diag(Qdiag)
        Rdiag = np.zeros(self.num_action)
        Rdiag[0] = 2.
        Rdiag[1] = 2.
        Rdiag[2] = 1.
        self.R = np.diag(Rdiag)


    def _run_controller(self):
        """ Main control loop """
        # Controller parameters

        horizon = 20

        # Set the current reference to the current positional estimate, at a
        # slight elevation
        # time.sleep(2)
        self.position_reference = [0., 0., 1.0]
        # Unlock the controller, BE CAREFLUE!!!
        self._cf.commander.send_setpoint(0, 0, 0, 0)

        state_data = []
        reference_data = []
        save_dir = 'data' # will cover the old ones
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            os.makedirs(save_dir + '/training_data')

        while True:

            timeStart = time.time()

            # Get the references
            x_r, y_r, z_r = self.position_reference
            dx_r, dy_r, dz_r = [0.0, 0.0, 0.0]
            target = np.array([x_r, y_r, z_r, dx_r, dy_r, dz_r])
            print("position_references", self.position_reference)

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
                lqr_policy = FiniteHorizonLQR(self.A, self.B, self.Q, self.R, self.Q, horizon = horizon)
                lqr_policy.set_target_state(target)  ## set target state to koopman observable state
                roll_r, pitch_r, thrust_r = lqr_policy(state)

                roll_r = self.saturate(roll_r * self.K_roll, self.roll_limit)
                pitch_r = self.saturate(pitch_r * self.K_pitch, self.pitch_limit)
                thrust_r = self.saturate(thrust_r * self.K_thrust + self.hover_thrust, self.thrust_limit)  # minus, see notebook
            else:
                # If the controller is disabled, send a zero-thrust
                roll_r, pitch_r, thrust_r = (0, 0, 0)
            # Communicate a reference value to the Crazyflie
            print("yaw angle: ", self.t.attitude[2])
            yaw_r = 0
            print("roll_r: ", roll_r)
            print("pitch_r: ", pitch_r)
            print("thrust_r: ", int(thrust_r))
            # self._cf.commander.send_setpoint(roll_r, - pitch_r, yaw_r, int(thrust_r)) # change!!!
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
                thrust_r = + self.saturate((Kzp * ez + Kzd * dez + self.m * self.g) * (self.hover_thrust/(self.m*self.g) / (cos(roll) * cos(pitch))),
                                           self.thrust_limit)
            else:
                # If the controller is disabled, send a zero-thrust
                roll_r, pitch_r, yaw_r, thrust_r = (0, 0, 0, 0)

            # Communicate a reference value to the Crazyflie
            self._cf.commander.send_setpoint(roll_r, pitch_r, 0, int(thrust_r))
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
            time.sleep(deltaTime)

    def set_reference(self, message):
        """ Enables an incremental change in the reference and defines the
        keyboard mapping (change to your preference, but if so, also make sure
        to change the valid_keys attribute in the interface thread)"""
        verbose = True
        if message == "s":
            self.position_reference[0] -= self.position_increments[0]
            if verbose: print('-x')
        if message == "w":
            self.position_reference[0] += self.position_increments[0]
            if verbose: print('+x')
        if message == "d":
            self.position_reference[1] -= self.position_increments[1]
            if verbose: print('-y')
        if message == "a":
            self.position_reference[1] += self.position_increments[1]
            if verbose: print('+y')
        if message == "k":
            self.position_reference[2] -= self.position_increments[2]
            if verbose: print('-z')
        if message == "i":
            self.position_reference[2] += self.position_increments[2]
            if verbose: print('+z')
        if message == "j":
            self.yaw_reference += self.yaw_increment
        if message == "l":
            self.yaw_reference -= self.yaw_increment
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