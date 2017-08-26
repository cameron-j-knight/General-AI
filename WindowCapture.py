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
        im = ImageGrab.grab(bbox=(self.window_size[0],
                                  self.window_size[1],
                                  self.window_size[0] + self.window_size[2],
                                  self.window_size[1] + self.window_size[3]))
        return im

    def vbGrab(self):
        vb = virtualbox.VirtualBox()
        vb.find_machine("Ubuntu")


if __name__ == '__main__':

    # #Test Run Efficeiency
    # window = WindowCapture()
    #
    # start_time = timeit.default_timer()
    #
    # # for i in range(100):
    # #     window.screen_grab()
    # window.screen_grab()
    # end_time = timeit.default_timer()
    #
    # window.screen_grab().show()
    #
    # print("time to run = ", end_time-start_time)

    window = WindowCapture()

    window.vbGrab()
