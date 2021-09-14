from networks import DynamicsModel
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import random
import numpy as np


class MBEnsemble():
    def __init__(self, state_size, action_size, config, device):
        
        self.device = device
        self.ensemble = []
        parameter = []
        for i in range(config.ensembles):
            dynamics = DynamicsModel(state_size=state_size,
                                               action_size=action_size,
                                               hidden_size=config.hidden_size,
                                               seed=i).to(device)
            self.ensemble.append(dynamics)
            parameter += list(dynamics.parameters())
            
        self.optimizer = optim.Adam(params=parameter, lr=config.mb_lr)
        
        self.n_updates = config.n_updates
        self.n_rollouts = config.n_rollouts
        self.kstep = config.kstep
        
        self.loss = nn.MSELoss()
        
    def train(self, dataloader):
        for epoch in range(self.n_updates):
            epoch_losses = []
            model = random.sample(self.ensemble, k=1)[0]
            for (s, a, r, ns, d) in dataloader:
                self.optimizer.zero_grad()
                prediction = model(s,a)
                targets = torch.cat((ns,r), dim=-1)
                loss = self.loss(prediction.float(), targets.to(self.device))
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())
        reward_diff = (r - prediction.detach()).mean()
        return np.mean(epoch_losses), reward_diff.item()
    
    def do_rollouts(self, buffer, env_buffer, policy):
        model = random.sample(self.ensemble, k=1)[0]
        states, _, _, _, _ = env_buffer.sample(self.n_rollouts)
        states = states.cpu().numpy()
        for k in range(self.kstep):
            
            actions = policy.get_action(states)
            
            predictions = model(torch.from_numpy(states).float().to(self.device),
                                torch.from_numpy(actions).float().to(self.device))
            next_states = predictions[:, :-1].detach().cpu().numpy()
            rewards = predictions[:, -1].detach().cpu().numpy()
            dones = torch.zeros(rewards.shape)
            for (s, a, r, ns, d) in zip(states, actions, rewards, next_states, dones):
                buffer.add(s, a, r, ns, d)
            
            states = next_states
            
    