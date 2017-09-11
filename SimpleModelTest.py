import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from scipy.misc import imresize
import math
import torch.nn.init as init
HEIGHT_IN = 256
WIDTH_IN = 256

def add_helper(value, tryIndex):
    try:
        return value[tryIndex]
    except:
        return value


class Policy(nn.Module):

    def _calculate_conv_size(self,conv,in_size):
        return math.floor((add_helper(in_size,0) + 2 * add_helper(conv.padding,0) - add_helper(conv.dilation,0) * (add_helper(conv.kernel_size,0) - 1) -1 ) / add_helper(conv.stride,0) + 1), \
               math.floor((add_helper(in_size, 1) + 2 * add_helper(conv.padding, 1) - add_helper(conv.dilation, 1) * (add_helper(conv.kernel_size, 1) - 1) - 1) / add_helper(conv.stride, 1) + 1)

    def __init__(self):

        conv_output_height = HEIGHT_IN
        conv_output_width = WIDTH_IN

        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(3, 64,kernel_size=2)
        conv_output_height, conv_output_width = self._calculate_conv_size(self.conv1,(conv_output_height,conv_output_width))

        self.pool1 = nn.MaxPool2d(kernel_size=2)
        conv_output_height, conv_output_width = self._calculate_conv_size(self.pool1,(conv_output_height,conv_output_width))

        self.conv2 = nn.Conv2d(64, 16,kernel_size=2)
        conv_output_height, conv_output_width = self._calculate_conv_size(self.conv2,(conv_output_height,conv_output_width))

        self.pool2 = nn.MaxPool2d(kernel_size=2)
        conv_output_height, conv_output_width = self._calculate_conv_size(self.pool2,(conv_output_height,conv_output_width))

        # self.conv3 = nn.Conv2d(16, 32,kernel_size=5)
        # conv_output_height, conv_output_width = self._calculate_conv_size(self.conv3,(conv_output_height,conv_output_width))
        #
        # self.conv4 = nn.Conv2d(32, 16,kernel_size=5)
        # conv_output_height, conv_output_width = self._calculate_conv_size(self.conv4,(conv_output_height,conv_output_width))
        #
        # self.pool3 = nn.MaxPool2d(kernel_size=2)
        # conv_output_height, conv_output_width = self._calculate_conv_size(self.pool3,(conv_output_height,conv_output_width))

        self.dropout = nn.Dropout2d()
        self.fully_connected = nn.Linear(conv_output_width * conv_output_height * 16, 64)

        self.class1 = nn.Linear(64,128)
        self.class2 = nn.Linear(128,28)
        self.output = nn.Linear(28,4)


        init.xavier_uniform(self.conv1.weight)
        init.xavier_uniform(self.conv2.weight)
        init.xavier_uniform(self.class1.weight)
        init.xavier_uniform(self.class2.weight)
        init.xavier_uniform(self.output.weight)

        init.uniform(self.conv1.bias)
        init.uniform(self.conv2.bias)
        init.uniform(self.class1.bias)
        init.uniform(self.class2.bias)
        init.uniform(self.output.bias)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        # x = F.relu(self.pool3(self.conv4(self.conv3(x))))
        x = F.relu(self.fully_connected(x.view(x.size(0), -1)))
        x = F.relu(self.class1(x))
        x = F.relu(self.class2(x))
        x = F.relu(self.output(x))
        return x

if __name__ == '__main__':
    import numpy as np
    p = Policy().cuda()
    val = torch.FloatTensor(1,3,HEIGHT_IN,WIDTH_IN)
    print(val.size())
    print(p(Variable(val).cuda()).size())

