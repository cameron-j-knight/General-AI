"""
Author: Cameron Knight
Description: Enviroment for running Pytorch Models that take a Image input.
"""
import sys, os
os.environ["OPENAI_REMOTE_VERBOSE"] = "0"

#imports
import gym
import universe  # register the universe environments
from universe import wrappers
from collections import namedtuple
from InPlace import *
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from scipy.misc import imresize
import torch
from torch import  optim
from torch.autograd import Variable
from torch import autograd
import torch.nn.functional as F
import torch.nn as nn
import logging
import Memory
import optimizers
from utils import *

#Policies
import SimpleModelTest
#Hyperparameters
batch_size = 5
iterations = 100000
max_lifetime = 2000
memory_size = 2000
learning_rate = 1e-5
gamma = .97


def move_from_prob(probs):
    """
    Creates an action from probabilities (output of the network)
    :param probs: outputs from the network softmax outputs
    :return: an action to be parsed by the network
    """
    _, action = torch.max(probs,1)
    action = action.data[0]
    act = [('KeyEvent', 'left', False),('KeyEvent', 'right', False), ('KeyEvent', 'up', False)]
    if(action == 0):
        act = [('KeyEvent', 'left', False), ('KeyEvent', 'right', False), ('KeyEvent', 'up', True)]
    elif(action == 1):
        act = [('KeyEvent', 'left', False), ('KeyEvent', 'right', True), ('KeyEvent', 'up', True)]
    elif(action == 3):
        act = [('KeyEvent', 'left', False), ('KeyEvent', 'right', False), ('KeyEvent', 'up', False)]
    return act


REWARD_BOOST_THRESHOLD = 15 #reward incease minimum per min to increase to kill bad subjects

def train():
    env = gym.make('flashgames.HeatRushUsa-v0') #gym.make('internet.SlitherIO-v0')

    #ensure all input is valid and vision is constrained to visible area
    env = wrappers.experimental.SafeActionSpace(env)
    env = wrappers.experimental.CropObservations(env)

    env.configure(remotes=1)
    observation = env.reset()

    policy = SimpleModelTest.Policy().cuda()

    #update to use only relevant params
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    # Graphing
    # plt.ion()
    # x_dat = []
    # y_dat = []
    # plt.show()

    # hl, = plt.plot(x_dat,y_dat)

    for i in range(iterations):
        reward_met = 0
        reward_met_iterations = 30
        memory = Memory(memory_size)
        started = False
        total_reward = 0
        for frames in range(max_lifetime):
            if not None in observation:
                vision = observation[0]['vision']

                vision_tensor = image_to_tensor(vision)
                action_probabilities, enc = policy(Variable(vision_tensor).cuda())

                action = [move_from_prob(action_probabilities)]

                observation, reward, done, info = env.step(action)
                total_reward += reward[0]
                action_tensor = torch.max(action_probabilities.data,1)[1]
                reward_tensor = torch.FloatTensor(reward)
                memory.push(vision_tensor, action_tensor, reward_tensor)
                #Check if stuff is happening
                # reward_met += reward[0]
                # if reward_met_iterations == 0:
                #     if reward_met/30 < REWARD_BOOST_THRESHOLD:
                #         done = True
                #     else:
                #         reward_met = 0
                #         reward_met_iterations = 30


            else:
                # Not ready to accept model inputs
                observation, reward, done, info = env.step([[] for _ in observation])


            if done:
                if started:
                    break
            else:
                started = True

            env.render()
        print("Total Reward:",total_reward)
        # x_dat += [i]
        # y_dat += [total_reward]
        #update_line(hl, x_dat, y_dat)
        for i in range(math.ceil(frames * total_reward//100) + 10 if math.ceil(frames* total_reward//100) + 10 < 150 else 150):
            optimizers.optimize_model(policy, optimizer, memory, batch_size, gamma)
        env.reset()

def update_line(hl, x, y):
    """
    updates dynamic graph if enabled
    :param hl: line
    :param x: new x value to add
    :param y: new y value to add
    :return: None
    """
    hl.set_xdata(x)
    hl.set_ydata(y)
    plt.draw()
    plt.pause(.1)


if __name__ == '__main__':
    #silence log
    train()
    # env = gym.make('flashgames.HeatRushUsa-v0') #gym.make('internet.SlitherIO-v0')
    #
    # #ensure all input is valid and vision is constrained to visible area
    # env = wrappers.experimental.SafeActionSpace(env)
    # env = wrappers.experimental.CropObservations(env)
    #
    # env.configure(remotes=1)
    # observation = env.reset()
    #
    # act = [('KeyEvent', 'left', False), ('KeyEvent', 'right', False), ('KeyEvent', 'up', True)]
    # while(True):
    #     for i in range(2000):
    #         env.step([act])
    #         env.render()
    #     env.reset()

