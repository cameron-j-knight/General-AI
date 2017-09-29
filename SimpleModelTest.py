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


def add_helper(value, tryIndex):
    """
    checks if value is indexed to return indexed value
    :param value: value to test
    :param tryIndex: index to try
    :return:
    """
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
        # Encoder
        self.conv1 = nn.Conv2d(3, 64,kernel_size=4)
        conv_output_height, conv_output_width = self._calculate_conv_size(self.conv1,(conv_output_height,conv_output_width))

        self.pool1 = nn.MaxPool2d(kernel_size=2, return_indices=True)
        conv_output_height, conv_output_width = self._calculate_conv_size(self.pool1,(conv_output_height,conv_output_width))

        self.conv2 = nn.Conv2d(64, 64,kernel_size=4)
        conv_output_height, conv_output_width = self._calculate_conv_size(self.conv2,(conv_output_height,conv_output_width))

        self.pool2 = nn.MaxPool2d(kernel_size=2, return_indices=True)
        conv_output_height, conv_output_width = self._calculate_conv_size(self.pool2,(conv_output_height,conv_output_width))

        self.conv3 = nn.Conv2d(64, 32,kernel_size=4)
        conv_output_height, conv_output_width = self._calculate_conv_size(self.conv3,(conv_output_height,conv_output_width))

        self.conv4 = nn.Conv2d(32, 16,kernel_size=4)
        conv_output_height, conv_output_width = self._calculate_conv_size(self.conv4,(conv_output_height,conv_output_width))

        self.pool3 = nn.MaxPool2d(kernel_size=2, return_indices=True)
        conv_output_height, conv_output_width = self._calculate_conv_size(self.pool3,(conv_output_height,conv_output_width))

        self.fully_connected = nn.Linear(conv_output_width * conv_output_height * 16, 4)

        self.encoded = nn.Linear(conv_output_width * conv_output_height * 16, 512)

        self.reconnect = nn.Linear(512,conv_output_width * conv_output_height * 16)

        # # Classification
        # self.class1 = nn.Linear(512,128)
        # self.class2 = nn.Linear(128,28)
        # self.output = nn.Linear(28,4)



        def deconv_from_conv(conv, kernal_size):
            return nn.ConvTranspose2d(in_channels=conv.out_channels,
                                      out_channels=conv.in_channels,
                                      kernel_size=kernal_size,
                                      stride=1,
                                      output_padding=0,
                                      )
        #Decoder
        self.unpool3 = nn.MaxUnpool2d(2)
        self.deconv4 = deconv_from_conv(self.conv4,5)
        self.deconv3 = deconv_from_conv(self.conv3,4)
        self.unpool2 = nn.MaxUnpool2d(2)
        self.deconv2 = deconv_from_conv(self.conv2,5)
        self.unpool1 = nn.MaxUnpool2d(2)
        self.deconv1 = deconv_from_conv(self.conv1,5)

        # self.autoencoder =  [self.conv1, self.conv2, self.conv3, self.conv4, self.pool1, self.pool2, self.pool3,
        #                     self.deconv1,self.deconv2, self.deconv3, self.deconv4, self.unpool1, self.unpool2,
        #                      self.unpool3]
        #
        # self.classifier = [self.conv1, self.conv2, self.conv3, self.conv4, self.pool1, self.pool2, self.pool3,
        #                    self.class1, self.class2, self.output]


        # Initialization
        # init.xavier_uniform(self.conv1.weight)
        # init.xavier_uniform(self.conv2.weight)
        # init.xavier_uniform(self.class1.weight)
        # init.xavier_uniform(self.class2.weight)
        # init.xavier_uniform(self.output.weight)
        #
        # init.uniform(self.conv1.bias)
        # init.uniform(self.conv2.bias)
        # init.uniform(self.class1.bias)
        # init.uniform(self.class2.bias)
        # init.uniform(self.output.bias)


    def forward(self, x):
        # conv
        conv1_size = x.size()
        x = self.conv1(x)
        x, pool1_indicies = self.pool1(x)
        x = F.relu(x)
        conv2_size = x.size()
        x = self.conv2(x)
        x, pool2_indicies = self.pool2(x)
        x = F.relu(x)

        conv3_size = x.size()
        x = self.conv3(x)
        conv4_size = x.size()
        x = self.conv4(x)
        x, pool3_indicies = self.pool3(x)
        x = F.relu(x)
        encoded_size = x.size()
        y = self.encoded(x.view(x.size(0), -1))
        x = self.fully_connected(x.view(x.size(0), -1))

        #deconv
        y = self.reconnect(y).view(encoded_size)
        y = F.relu(self.deconv3(self.deconv4(self.unpool3(y,pool3_indicies),output_size=conv4_size),output_size=conv3_size))
        y = F.relu(self.deconv2(self.unpool2(y,pool2_indicies),output_size=conv2_size))
        y = F.relu(self.deconv1(self.unpool1(y,pool1_indicies),output_size=conv1_size))

        #classification
        # x = F.relu(self.class1(x))
        # x = F.relu(self.class2(x))
        # x = F.relu(self.output(x))
        return x,y


HEIGHT_IN = 256
WIDTH_IN = 256
if __name__ == '__main__':
    import numpy as np
    batch_size = 5
    p = Policy().cuda()
    optimizer = optim.Adam(p.parameters(),lr=.1)



    for i in range(10000):
        a = Variable(torch.FloatTensor(np.random.uniform(0,1,(batch_size,3,HEIGHT_IN,WIDTH_IN)))).cuda()
        o, auto_encode = p(a)
        loss =  torch.sum(F.cosine_similarity(auto_encode,a))
        print(loss.size())
        loss.backward()
        # print("outs:", o)
        print("step:",i,"loss:",loss.data[0]/(256**2))
        # print('+'*20)
        optimizer.step()
