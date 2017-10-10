from collections import namedtuple
from InPlace import InPlaceArray
import random

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
        samp = random.sample([i for i in self.memory if i.future_state[-1].value is not None], batch_size)
        # while(len([True for i in samp if i.future_state[-1] is not None])):
        #     samp = random.sample(self.memory, batch_size)
        return samp

    def __len__(self):
        return len(self.memory)

    def __str__(self):

        return str(self.memory)
