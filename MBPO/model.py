from networks import DynamicsModel
import torch
import torch.optim as optim
import random
import numpy as np


def termination_fn(env_name, obs, act, next_obs, rewards):
    if env_name == "HopperBulletEnv-v":
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        height = next_obs[:, 0]
        angle = next_obs[:, 1]
        not_done =  np.isfinite(next_obs).all(axis=-1) \
                    * np.abs(next_obs[:,1:] < 100).all(axis=-1) \
                    * (height > .7) \
                    * (np.abs(angle) < .2)

        done = ~not_done
        done = done[:,None]
        return done
    elif env_name == "Walker2dBulletEnv-v0":
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        height = next_obs[:, 0]
        angle = next_obs[:, 1]
        not_done =  (height > 0.8) \
                    * (height < 2.0) \
                    * (angle > -1.0) \
                    * (angle < 1.0)
        done = ~not_done
        done = done[:,None]
        return done
    else:
        done = torch.zeros(rewards.shape).bool()
        return done
        

class MBEnsemble():
    def __init__(self, state_size, action_size, config, device):
                
        self.device = device
        self.ensemble = []

        self.n_ensembles = config.ensembles
        for i in range(self.n_ensembles):
            dynamics = DynamicsModel(state_size=state_size,
                                               action_size=action_size,
                                               hidden_size=config.hidden_size,
                                               lr=config.mb_lr,
                                               seed=i,
                                               device=device).to(device)
            self.ensemble.append(dynamics)          

        self.n_rollouts = config.n_rollouts * config.parallel_envs
        self.rollout_select = config.rollout_select
        self.elite_size = config.elite_size
        self.elite_idxs = []
        
        self.env_name = config.env
        
    def train(self, train_dataloader):
        epoch_losses = []
        for model in self.ensemble:
            for (s, a, r, ns, d) in train_dataloader:
                delta_state = ns - s
                targets = torch.cat((delta_state, r), dim=-1).to(self.device)
                loss = model.calc_loss(s, a, targets)
                model.optimize(loss)

            
            epoch_losses.append(loss.item())
        assert len(epoch_losses) == self.n_ensembles, f"epoch_losses: {len(epoch_losses)} =/= {self.n_ensembles}"
        sorted_loss_idx = np.argsort(epoch_losses)
        self.elite_idxs = sorted_loss_idx[:self.elite_size].tolist()
        
        return np.mean(epoch_losses)
    
    def run_ensemble_prediction(self, states, actions):
        prediction_list = []
        with torch.no_grad():
            for model in self.ensemble:
                predictions, _ = model(torch.from_numpy(states).float().to(self.device),
                                        torch.from_numpy(actions).float().to(self.device))
                prediction_list.append(predictions.unsqueeze(0))
        all_ensemble_predictions = torch.cat(prediction_list, axis=0) 
        # [ensembles, batch, prediction_shape]
        assert all_ensemble_predictions.shape == (self.n_ensembles, states.shape[0], states.shape[1] + 1)
        return all_ensemble_predictions


    def do_rollouts(self, buffer, env_buffer, policy, kstep):
        
        states, _, _, _, _ = env_buffer.sample(self.n_rollouts)
        states = states.cpu().numpy()
        for k in range(kstep):
            actions = policy.get_action(states)
            all_ensemble_predictions = self.run_ensemble_prediction(states, actions)
            if self.rollout_select == "random":
                # choose what predictions we select from what ensemble member
                ensemble_idx = random.choices(self.elite_idxs, k=self.n_rollouts)
                step_idx = np.arange(states.shape[0])
                # pick prediction based on ensemble idxs
                predictions = all_ensemble_predictions[ensemble_idx, step_idx, :]
            else:
                predictions = all_ensemble_predictions[self.elite_idxs].mean(0)
            assert predictions.shape == (self.n_rollouts, states.shape[1] + 1)
            delta_state = predictions[:, :-1].cpu().numpy()
            next_states = states + delta_state
            rewards = predictions[:, -1].cpu().numpy()
            dones = termination_fn(self.env_name, states, actions, next_states, rewards)
            for (s, a, r, ns, d) in zip(states, actions, rewards, next_states, dones):
                buffer.add(s, a, r, ns, d)
            if ~dones.sum() == 0:
                break
            states = next_states
        # calculate epistemic uncertainty ~ variance between the ensembles 
        # over the course of training ensembles should all predict the same variance -> 0 
        # model is very certain what will happen
        variance_over_each_state = all_ensemble_predictions.var(0)
        whole_batch_uncertainty = variance_over_each_state.var()
        return whole_batch_uncertainty.item()

    def value_expansion(self, rewards, next_state, policy, gamma=0.99):
        rollout_reward = np.zeros((rewards.shape))

        for h in range(self.mve_horizon):
            output_state = next_state
            rollout_reward += (gamma**h * rewards.cpu().numpy())
            action = policy.get_action(next_state)
            predictions = self.run_ensemble_prediction(next_state, action).mean(0)
            assert predictions.shape == (next_state.shape[0], next_state.shape[1]+1)
            next_state = predictions[:, :-1].cpu().numpy()
            rewards = predictions[:, -1].unsqueeze(-1)
        
        return torch.from_numpy(output_state).float().to(self.device), torch.from_numpy(rollout_reward).float().to(self.device)
            
    
