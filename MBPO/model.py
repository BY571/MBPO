from networks import DynamicsModel
import torch
import torch.optim as optim
import random
import numpy as np


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

        self.n_rollouts = config.n_rollouts
        self.rollout_select = config.rollout_select
        self.stop_early = 4
        self.elite_idxs = []
        
    def train(self, train_dataloader, test_dataloader):
        epoch_losses = []
        epochs_trained = []
        for model in self.ensemble:
            epoch = 0
            lowest_validation = np.inf
            stop_early_counter = 0
            while True:
                # train
                train_losses = []
                for (s, a, r, ns, d) in train_dataloader:
                    targets = torch.cat((ns,r), dim=-1).to(self.device)
                    loss = model.calc_loss(s, a, targets)
                    model.optimize(loss)
                    
                    epoch_losses.append(loss.item())
                    train_losses.append(loss.item())
                # evaluation
                model.eval()
                validation_losses = []
                for (s, a, r, ns, d) in test_dataloader:
                    with torch.no_grad():
                        targets = torch.cat((ns,r), dim=-1).to(self.device)
                        validation_loss = model.calc_loss(s, a, targets)
                    validation_losses.append(validation_loss.item())
                    
                model.train()
                epoch += 1

                if np.mean(validation_losses) < lowest_validation:
                    lowest_validation = np.mean(validation_losses)
                    stop_early_counter = 0 
                else:
                    stop_early_counter += 1
                if stop_early_counter >= self.stop_early:
                    # print("-- Stop early at epoch: {} --".format(epoch))
                    epochs_trained.append(epoch)
                    break
                

        return np.mean(epoch_losses), np.mean(epochs_trained)
    
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
                ensemble_idx = random.choices(range(len(self.ensemble)), k=self.n_rollouts)
                step_idx = np.arange(states.shape[0])
                # pick prediction based on ensemble idxs
                predictions = all_ensemble_predictions[ensemble_idx, step_idx, :]
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
            
    