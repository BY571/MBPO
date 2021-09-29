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
        self.mve_horizon = config.mve_horizon
        self.rollout_select = config.rollout_select
        
    def train(self, dataloader):
        epoch_losses = []
        for i in range(self.n_updates):
            for model in self.ensemble:
                for (s, a, r, ns, d) in dataloader:
                    self.optimizer.zero_grad()
                    prob_prediction, (mu, log_var), _ = model(s,a)
                    inv_var = (-log_var).exp()
                    targets = torch.cat((ns,r), dim=-1)
                    loss = ((mu - targets.to(self.device))**2 * inv_var).mean(-1).mean(-1) + log_var.mean(-1).mean(-1)
                    
                    loss.backward()
                    self.optimizer.step()
                    epoch_losses.append(loss.item())
            reward_diff = (r - prob_prediction[:, -1].detach()).mean()
        return np.mean(epoch_losses), reward_diff.item()
    
    def run_ensemble_prediction(self, states, actions):
        prediction_list = []
        with torch.no_grad():
            for model in self.ensemble:
                predictions, _, _ = model(torch.from_numpy(states).float().to(self.device),
                                        torch.from_numpy(actions).float().to(self.device))
                prediction_list.append(predictions.unsqueeze(0))
        all_ensemble_predictions = torch.cat(prediction_list, axis=0) 
        # [ensembles, batch, prediction_shape]
        return all_ensemble_predictions


    def do_rollouts(self, buffer, env_buffer, policy, kstep):
        
        states, _, _, _, _ = env_buffer.sample(self.n_rollouts)
        states = states.cpu().numpy()
        for k in range(kstep):
            actions = policy.get_action(states)
            all_ensemble_predictions = self.run_ensemble_prediction(states, policy)
            if self.rollout_select == "random":
                # choose what predictions we select from what ensemble member
                idxs = random.choices(range(len(self.ensemble)), k=self.n_rollouts)
                # pick prediction based on ensemble idxs
                predictions = all_ensemble_predictions[idxs, 1, :]
            else:
                predictions = all_ensemble_predictions.mean(0)
            assert predictions.shape == (self.n_rollouts, states.shape[1] + 1)
            next_states = predictions[:, :-1].cpu().numpy()
            rewards = predictions[:, -1].cpu().numpy()
            dones = torch.zeros(rewards.shape)
            for (s, a, r, ns, d) in zip(states, actions, rewards, next_states, dones):
                buffer.add(s, a, r, ns, d)
            states = next_states
        # calculate epistemic uncertainty ~ variance between the ensembles 
        # over the course of training ensembles should all predict the same variance -> 0 
        # model is very certain what will happen
        variance_over_each_state = all_ensemble_predictions.var(0)
        whole_batch_uncertainty = variance_over_each_state.var()
        return whole_batch_uncertainty.item()

    def value_expansion(self, rewards, next_state, policy, gamma=0.99):
        rollout_reward = 0

        for h in range(self.mve_horizon):
            output_state = next_state
            rollout_reward += (gamma**h * rewards)
            action = policy.get_action(next_state)
            predictions = self.run_ensemble_prediction(next_state.numpy(), action.numpy()).mean(0)
            assert predictions.shape == (next_state.shape[0], next_state.shape[1]+1)
            next_state = predictions[:, :-1].cpu().numpy()
            rewards = predictions[:, -1].cpu().numpy()
        
        return output_state, rollout_reward
            
    