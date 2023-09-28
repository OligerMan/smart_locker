import serial
from enum import Enum


class Command(Enum):
    NO_DATA = -1
    # working mode 1 - direct commands to 2 lockers
    LEFT_UP = 0
    LEFT_DOWN = 1
    RIGHT_UP = 2
    RIGHT_DOWN = 3
    # working mode 2 - switching between 4 lockers and open/close
    SELECT_LEFT_UP = 4
    SELECT_LEFT_DOWN = 5
    SELECT_RIGHT_UP = 6
    SELECT_RIGHT_DOWN = 7
    OPEN_CELL = 8
    CLOSE_CELL = 9
    OPEN_CELL_NO_PERMISSION = 10
    CLOSE_CELL_NO_PERMISSION = 11


class Arduino:
    def __init__(self, port=None, baudrate=9600, timeout=.01):
        if port is not None:
            self._serial = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        else:
            self._serial = None

    def send_command(self, command: Command):
        if self._serial is not None:
            self._serial.write(command.value.to_bytes(1, byteorder='big', signed=True))

    def get_data(self):
        if self._serial is None:
            return "arduino is not connected"
        return self._serial.readline()