"""
Author: Cameron Knight
Project:
framework for mapping a controller actions to physical locations to help
build relations between differenct controllers

"""

from PIL import Image,ImageDraw

class Button:
    """
    Button in a physical location of a size with an action that can be preformed
    """
    def __init__(self, action, location,size, color="#FFFFFF", name=""):
        #set button properties
        self.location = location
        self.size = size
        self.action = action

        self.name = name
        self.color = color

        self.pressed = False

    def press(self,location=(0,0)):
        if not self.pressed:
            self.action("began",location)
            self.pressed = True
        else:
            self.action("held",location)

    def release(self,location=(0,0)):
        if self.pressed:
            self.action("ended",location)
            self.pressed = False

    def handle_touch_locations(self,locations):
        is_touch_on_location = False
        relative_location = (0,0)
        for location in locations[::-1]:
            if(self.is_touch_on_button(location)):
                is_touch_on_location = True
                relative_x = ((location[0] - self.location[0]) / self.size[0])
                relative_y = ((location[1] - self.location[1]) / self.size[1])
                relative_location = (relative_x, relative_y)

        if is_touch_on_location:
            self.press(relative_location)
        else:
            self.release()

    def is_touch_on_button(self,location):
        if location[0] < self.location[0] - self.size[0] * 0.5:
            return False
        if location[0] > self.location[0] + self.size[0] * 0.5:
            return False
        if location[1] < self.location[1] - self.size[1] * 0.5:
            return False
        if location[1] > self.location[1] + self.size[1] * 0.5:
            return False
        return True


    def render(self, image):
        size = [0,0]
        size[0] = int(self.size[0] * image.size[0]);
        size[1] = int(self.size[1] * image.size[1]);

        location = [0,0]
        location[0] = int(self.location[0] * image.size[0] - size[0] * 0.5);
        location[1] = int(self.location[1] * image.size[1] - size[1] * 0.5);


        im = Image.new(mode='RGBA', size=size, color=self.color)

        image.paste(im,location)


class Controller:
    def __init__(self):
        self.buttons = []

    def add_button(self,button):
        self.buttons += [button]

    def press(self,locations):
        for button in self.buttons:
            button.handle_touch_locations(locations)

    def render(self, size=(512,512),background_color="#000000"):
        im = Image.new(mode= 'RGBA',size=size,color=background_color)
        for button in self.buttons:
            button.render(im)
        return im

class XBoxController(Controller):
    def _make_xbox_button(self,location,size):
        pass

    def __init__(self):
        super.__init__()

class KeyboardController(Controller):
    def __init__(self):
        super.__init__()


if __name__ == '__main__':

    test_controller = Controller()
    test_controller.add_button(Button(action=lambda x,y: print(x,y), location=(0.25,0.5), size=(0.15,0.5), color="#999999"))
    test_controller.add_button(Button(action=lambda x,y: print(x,y), location=(0.50,0.5), size=(0.15,0.5), color='#00FF00'))
    test_controller.add_button(Button(action=lambda x,y: print(x,y), location=(0.75,0.5), size=(0.15,0.5), color='#0000FF'))

    test_controller.press([(0.75,0.5), (0.50,0.50), (0.25,0.50)])
    test_controller.press([(0.75,0.5), (0.30,0.50)])
    test_controller.press([(0.75,0.5), (0.50,0.50), (0.25,0.50)])
    test_controller.press([(0,0),(0,0),(0,0)])

    # test_controller.press([(0.15,0.15)])
    test_controller.render().show()
