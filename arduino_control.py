import serial
from enum import Enum


class Command(Enum):
    NO_DATA = -1
    LEFT_UP = 0
    LEFT_DOWN = 1
    RIGHT_UP = 2
    RIGHT_DOWN = 3


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