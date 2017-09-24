"""
Author: Cameron Knight
Description: Basic window capture script for capturing any window on any OS.
Quite slow probably need to use virtual machine enviroment to do more efficient calculations
"""

#imports
from PIL import Image
import pyscreenshot as ImageGrab
import os
import time
import subprocess
import timeit

import virtualbox


class WindowCapture():
    """
    capturable window framework
    """
    setup_time = 3

    def _get_window_size(self):
        """
        gets active window size by requesting user to select window after selected time
        :return: a 4 tuple of data (absolute upper left x, absolute upper left y width, height)
        """
        print("Open active window: \n starting in:")
        for i in range(self.setup_time,0,-1):
            print(i)
            time.sleep(1)

        output = subprocess.getoutput("xwininfo")

        properties = {}
        for line in output.split("\n"):
            if ":" in line:
                parts = line.split(":", 1)
                properties[parts[0].strip()] = parts[1].strip()
        return (float(properties["Absolute upper-left X"]),
                float(properties["Absolute upper-left Y"]),
                float(properties["Width"]),
                float(properties["Height"]))

    def __init__(self):
        self._screen_dims = ImageGrab.grab().size
        # self.window_size = self._get_window_size()

    def screen_grab(self):
        """
        grabs the current selected window
        :return: PIL image of current window selected
        """
        im = ImageGrab.grab(bbox=(self.window_size[0],
                                  self.window_size[1],
                                  self.window_size[0] + self.window_size[2],
                                  self.window_size[1] + self.window_size[3]))
        return im


if __name__ == '__main__':

    # Test Run Efficeiency
    window = WindowCapture()

    start_time = timeit.default_timer()

    for i in range(100):
        window.screen_grab()
    end_time = timeit.default_timer()

    # Show what was grabbed
    window.screen_grab().show()

    print("time to run = ", end_time-start_time)
