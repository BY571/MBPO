from networks import DynamicsModel
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Normal
import random
import numpy as np


class MBEnsemble():
    def __init__(self, state_size, action_size, config, device):
                
        self.device = device
        self.ensemble = []
        parameter = []
        self.n_ensembles = config.ensembles
        for i in range(self.n_ensembles):
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
        
        # self.loss = nn.MSELoss()
        # self.loss = nn.GaussianNLLLoss()
        
    def train(self, dataloader):
        for epoch in range(self.n_updates):
            epoch_losses = []
            model = random.sample(self.ensemble, k=1)[0]
            for (s, a, r, ns, d) in dataloader:
                self.optimizer.zero_grad()
                prob_prediction, (mu, log_var), _ = model(s,a)
                inv_var = (-log_var).exp()
                targets = torch.cat((ns,r), dim=-1)
                #loss = - dist.log_prob(targets).mean()
                loss = ((mu - targets.to(self.device))**2 * inv_var).mean(-1).mean(-1) + log_var.mean(-1).mean(-1)
                
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())
        reward_diff = (r - prob_prediction[:, -1].detach()).mean()
        return np.mean(epoch_losses), reward_diff.item()
    
    def do_rollouts(self, buffer, env_buffer, policy):
        
        states, _, _, _, _ = env_buffer.sample(self.n_rollouts)
        states = states.cpu().numpy()
        for k in range(self.kstep):
            #model = random.sample(self.ensemble, k=1)[0]
            actions = policy.get_action(states)
            ensemble_mu = 0
            ensemble_log_var = 0
            with torch.no_grad():
                for model in self.ensemble:
                    _, (mu, log_var), _ = model(torch.from_numpy(states).float().to(self.device),
                                            torch.from_numpy(actions).float().to(self.device))
                    ensemble_mu += mu
                    ensemble_log_var += log_var.exp()

            dist = Normal(ensemble_mu/self.n_ensembles,
                          ensemble_log_var/self.n_ensembles)
            predictions = dist.sample()
                
            next_states = predictions[:, :-1].cpu().numpy()
            rewards = predictions[:, -1].cpu().numpy()
            dones = torch.zeros(rewards.shape)
            for (s, a, r, ns, d) in zip(states, actions, rewards, next_states, dones):
                buffer.add(s, a, r, ns, d)
            
            states = next_states
            
    