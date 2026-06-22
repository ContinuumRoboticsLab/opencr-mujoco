import numpy as np
import atexit
import time
from pydualsense import pydualsense
from pydualsense.enums import PlayerID


class DualSenseInputDevice:
    """Class for connecting to and reading from PS5 DualSense Controller."""

    def __init__(self):
        self.controller = pydualsense()
        self.controller.init()
        self._closed = False
        self.controller.light.setColorT((255, 255, 255))
        self.controller.share_pressed += self.share_down
        self.controller.ps_pressed += self.ps_down

        self.controller.triangle_pressed += self.triangle_down
        self.controller.cross_pressed += self.cross_down
        self.controller.square_pressed += self.square_down
        self.controller.circle_pressed += self.circle_down

        self.controller.dpad_up += self.dpad_up_pressed
        self.controller.dpad_down += self.dpad_down_pressed
        self.controller.dpad_left += self.dpad_left_pressed
        self.controller.dpad_right += self.dpad_right_pressed

        self.controller.left_joystick_changed += self.ljoystick
        self.controller.right_joystick_changed += self.rjoystick
        self.controller.l1_changed += self.l1
        self.controller.r1_changed += self.r1

        # Add gyroscope and accelerometer callbacks
        self.controller.gyro_changed += self.gyro_changed
        self.controller.accelerometer_changed += self.accelerometer_changed

        self.ps_down_state = False
        self.share_down_state = False
        self.triangle_down_state = False
        self.cross_down_state = False
        self.square_down_state = False
        self.circle_down_state = False
        self.dpad_up_state = False
        self.dpad_down_state = False
        self.dpad_left_state = False
        self.dpad_right_state = False
        self.right_joystick_state = np.array([0.0, 0.0])
        self.left_joystick_state = np.array([0.0, 0.0])
        self.l1_state = False
        self.r1_state = False

        # Add gyroscope and accelerometer state
        self.gyro_state = np.array([0.0, 0.0, 0.0])  # x, y, z angular velocity
        self.accel_state = np.array([0.0, 0.0, 0.0])  # x, y, z linear acceleration

        atexit.register(self.close)

    def ps_down(self, state):
        self.ps_down_state = state

    def share_down(self, state):
        self.share_down_state = state

    def triangle_down(self, state):
        self.triangle_down_state = state

    def cross_down(self, state):
        self.cross_down_state = state

    def square_down(self, state):
        self.square_down_state = state

    def circle_down(self, state):
        self.circle_down_state = state

    def dpad_up_pressed(self, state):
        self.dpad_up_state = state

    def dpad_down_pressed(self, state):
        self.dpad_down_state = state

    def dpad_left_pressed(self, state):
        self.dpad_left_state = state

    def dpad_right_pressed(self, state):
        self.dpad_right_state = state

    def rjoystick(self, stateX, stateY):
        self.right_joystick_state = np.array([stateX, stateY])

    def ljoystick(self, stateX, stateY):
        self.left_joystick_state = np.array([stateX, stateY])

    def l1(self, state):
        self.l1_state = state

    def r1(self, state):
        self.r1_state = state

    def gyro_changed(self, x, y, z):
        """Callback for gyroscope data changes"""
        self.gyro_state = np.array([float(x), float(y), float(z)])

    def accelerometer_changed(self, x, y, z):
        """Callback for accelerometer data changes"""
        self.accel_state = np.array([float(x), float(y), float(z)])

    def set_light_playerid(self, player_id: int):
        enums = [
            PlayerID.PLAYER_1,
            PlayerID.PLAYER_2,
            PlayerID.PLAYER_3,
            PlayerID.PLAYER_4,
        ]
        self.controller.light.setPlayerID(enums[player_id - 1])

    def close(self):
        if not self._closed:
            self.controller.light.setColorT((0, 0, 0))
            time.sleep(0.01)
            self.controller.close()
            self._closed = True


def main():
    """Test the DualSense controller is connected and working"""
    DualTest = DualSenseInputDevice()
    for _ in range(100):
        print(
            f"Joysticks: {DualTest.right_joystick_state}, {DualTest.left_joystick_state}"
        )
        print(f"Gyro: {DualTest.gyro_state}, Accel: {DualTest.accel_state}")
        print(f"Cross: {DualTest.cross_down_state}")
        time.sleep(0.01)
    DualTest.close()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
