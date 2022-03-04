import numpy as np
import random
import torch
from collections import deque, namedtuple
from operator import itemgetter

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self, samples=None):
        """Randomly sample a batch of experiences from memory."""
        if samples == None:
            experiences = random.sample(self.memory, k=self.batch_size)
        else:
            experiences = random.sample(self.memory, k=samples)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class MBReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)
        self.position = 0
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    def sample(self, samples=None):
        """Randomly sample a batch of experiences from memory."""

        states = torch.from_numpy(np.stack([e.state for e in self.memory if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in self.memory if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in self.memory if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in self.memory if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in self.memory if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states)

    def sample_random_state(self, samples):
        idxes = np.random.randint(0, len(self.memory), samples)
        experiences = list(itemgetter(*idxes)(self.memory))
        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        return states
        
    def return_all(self,):
        return self.memory
        
    def push_batch(self, batch):
        for i in batch: self.memory.append(i)
        
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

