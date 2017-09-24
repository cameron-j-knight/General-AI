"""
Author: Cameron Knight
Description: Enviroment for running Pytorch Models that take a Image input.
"""

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
from torch import optim
from torch.autograd import Variable
from torch import autograd
import torch.nn.functional as F
import torch.nn as nn

#Policies
import SimpleModelTest

#Hyperparameters
batch_size = 10
iterations = 1000
max_lifetime = 2000
memory_size = 2000
learning_rate = 1e-2

Transition = namedtuple('Transition',
                        ('state', 'action', 'future_state', 'reward'))

class Memory():
    def __init__(self, capacity=1000,future_events=1,future_spacing=1):
        """
        initializes a memory unit
        :param capacity: The number of frames to remember
        :param future_events: number of events to remember into the future
        :param future_spacing: the number of frames in the future in which that event takes place
        """
        self.capacity = capacity
        self.memory = []
        self.states = InPlaceArray(*[None for _ in range(capacity)])
        self.position = 0
        self.future_events = future_events
        self.future_spacing = future_spacing

    def push(self, state, action, reward):
        """Saves a transition."""
        self.states[self.position] = state
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        future_events = [self.states[event] for event in range((self.position + 1) % self.capacity,
                                        (self.position + 1)% self.capacity + self.future_spacing * self.future_events,
                                        self.future_spacing)]

        self.memory[self.position] = Transition(self.states[self.position], action, future_events, reward)

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def __str__(self):

        return str(self.memory)



def optimize_model(model, optimizer, memory):
    print('optimizing...')
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))


    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.ByteTensor(tuple(map(lambda s: s[-1].value is not None, batch.future_state))).cuda()
    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = Variable(torch.cat([s[-1].value for s in batch.future_state
                                                if s[-1].value is not None]),
                                     volatile=True).cuda()
    state_batch = Variable(torch.cat([s.value for s in batch.state])).cuda()
    action_batch = Variable(torch.cat(batch.action)).cuda()
    reward_batch = Variable(torch.cat(batch.reward)).cuda()
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch.view(-1,1)).cuda()

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(non_final_mask.size())).cuda()
    next_state_values[non_final_mask] = model(non_final_next_states)
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * .99) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    #Clamp gradients between -1 and 1
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def image_to_tensor(img, size=256):
    """
    Converts an image array into a Tensor of fixed size
    :param img: image to transpose
    :return: image tensor size x size
    """
    state = imresize(img,[size,size], 'nearest')

    input_tensor = torch.FloatTensor(state.reshape(-1,3,size,size)/255.0)

    return input_tensor


def move_from_prob(probs):
    """
    Creates an action from probabilities (output of the network)
    :param probs: outputs from the network softmax outputs
    :return: an action to be parsed by the network
    """
    _, action = torch.max(probs,1)
    action = action[0]
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
                action_probabilities = policy(Variable(vision_tensor).cuda()).data

                action = [move_from_prob(action_probabilities)]

                observation, reward, done, info = env.step(action)
                total_reward += reward[0]
                action_tensor = torch.max(action_probabilities,1)[1]
                reward_tensor = torch.FloatTensor(reward)
                memory.push(vision_tensor, action_tensor, reward_tensor)
                #Check if stuff is happening
                reward_met += reward[0]
                if reward_met_iterations == 0:
                    if reward_met/30 < REWARD_BOOST_THRESHOLD:
                        done = True
                    else:
                        reward_met = 0
                        reward_met_iterations = 30


            else:
                # Not ready to accept model inputs
                observation, reward, done, info = env.step([[] for _ in observation])

            env.render()

            if done:
                if started:
                    break
            else:
                started = True
        print("Total Reward:",total_reward)
        # x_dat += [i]
        # y_dat += [total_reward]
        #update_line(hl, x_dat, y_dat)
        for i in range(math.ceil(frames * total_reward//100) + 10 if math.ceil(frames* total_reward//100) + 10 < 150 else 150):
            optimize_model(policy,optimizer,memory)
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
    train()

