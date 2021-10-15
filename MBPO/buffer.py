import numpy as np
import random
import torch
from collections import deque, namedtuple
from torch.utils.data import TensorDataset, DataLoader
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

        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        for (s, a, r, ns, d) in zip(state, action, reward, next_state, done):
            e = self.experience(s, a, r, ns, d)
            self.memory.append(e)
        
    def sample(self, samples=None):
        """Randomly sample a batch of experiences from memory."""
        idxes = np.random.randint(0, len(self.memory), samples)
        experiences = list(itemgetter(*idxes)(self.memory))
        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def get_dataloader(self, batch_size=256):
        states = torch.from_numpy(np.stack([e.state for e in self.memory if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in self.memory if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in self.memory if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in self.memory if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in self.memory if e is not None]).astype(np.uint8)).float().to(self.device)
        dataset = TensorDataset(states, actions, rewards, next_states, dones)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return dataloader
    
    def return_all(self,):
        return self.memory
    
    def push_batch(self, batch):
        if len(self.memory) < self.buffer_size:
            append_len = min(self.buffer_size - len(self.memory), len(batch))
            self.memory.extend([None] * append_len)

        if self.position + len(batch) < self.buffer_size:
            self.memory[self.position : self.position + len(batch)] = batch
            self.position += len(batch)
        else:
            self.memory[self.position : len(self.memory)] = batch[:len(self.memory) - self.position]
            self.memory[:len(batch) - len(self.memory) + self.position] = batch[len(self.memory) - self.position:]
            self.position = len(batch) - len(self.memory) + self.position
        
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

