import gym
import universe  # register the universe environments
from universe import wrappers

from collections import namedtuple

from InPlace import *

import numpy as np

import random

from scipy.misc import imresize
from scipy.misc import toimage

import torch
from torch import optim
from torch.autograd import Variable
from torch import autograd
import torch.nn.functional as F
import torch.nn as nn

#Policies
import SimpleModelTest

#Hyperparameters
batch_size = 100

Transition = namedtuple('Transition',
                        ('state', 'action', 'future_state', 'reward'))

class Memory():
    def __init__(self, capacity=500,future_events=1,future_spacing=1):
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
        self.states += [state]
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        future_events = [self.states[event] for event in range(self.position + 1,
                                              self.position + 1 + self.future_spacing * self.future_events,
                                              self.future_spacing)]

        self.memory[self.position] = Transition(self.states[self.position], action, future_events, reward)

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



def optimize_model(model, optimizer, memory):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.ByteTensor(tuple(map(lambda s: s[-1].value is not None, batch.future_state)))

    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = Variable(torch.cat([s[-1] for s in batch.future_state
                                                if s[-1] is not None]),
                                     volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(20))
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
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
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def flush_policy(model):
    del model.rewards[:]

def image_to_tensor(img):
    state = imresize(img,[256,256], 'nearest')

    input_tensor = torch.FloatTensor(state.reshape(1,3,256,256)/255.0)

    return input_tensor


def move_from_prob(probs):
    _, action = torch.max(probs,1)
    action = action[0]
    print(probs)
    act = [('KeyEvent', 'left', True),('KeyEvent', 'right', False), ('KeyEvent', 'up', True)]
    if(action == 0):
        act = [('KeyEvent', 'left', True), ('KeyEvent', 'right', False), ('KeyEvent', 'up', True)]
    elif(action == 1):
        act = [('KeyEvent', 'left', False), ('KeyEvent', 'right', True), ('KeyEvent', 'up', True)]
    elif(action == 3):
        act = [('KeyEvent', 'left', False), ('KeyEvent', 'right', False), ('KeyEvent', 'up', True)]
    else:
        act = [('KeyEvent', 'left', False), ('KeyEvent', 'right', False), ('KeyEvent', 'up', False)]
    # model.saved_actions.append(SavedAction(action))
    return act


REWARD_BOOST_THRESHOLD = 5 #reward incease minimum per min to increase to kill bad subjects


def train():
    env = gym.make('flashgames.HeatRushUsa-v0') #gym.make('internet.SlitherIO-v0')

    #ensure all input is valid and vision is constrained to visible area
    env = wrappers.experimental.SafeActionSpace(env)
    env = wrappers.experimental.CropObservations(env)

    env.configure(remotes=1)
    observation = env.reset()

    policy = SimpleModelTest.Policy().cuda()
    optimizer = optim.RMSprop(policy.parameters(), lr=1e-2)

    for i in range(100):
        reward_met = 0
        reward_met_iterations = 30
        memory = Memory(500)
        started = False

        for i in range(500):
            if not None in observation:
                vision = observation[0]['vision']

                vision_tensor = image_to_tensor(vision)
                action_probabilities = policy(Variable(vision_tensor).cuda()).data

                action = [move_from_prob(action_probabilities)]

                observation, reward, done, info = env.step(action)
                memory.push(vision_tensor, action[0], reward[0])
                #Check if stuff is happening
                reward_met += reward[0]
                if reward_met_iterations == 0:
                    if reward_met < REWARD_BOOST_THRESHOLD:
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
        env.reset()
        # optimize_model(policy,optimizer,memory)

def test_iter():
    pass


if __name__ == '__main__':
    train()
    pass

