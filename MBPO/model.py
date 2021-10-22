from networks import DynamicsModel
import torch
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

        self.probabilistic = True
        self.n_ensembles = config.ensembles

        self.dynamics_model = DynamicsModel(state_size=state_size,
                                            action_size=action_size,
                                            ensemble_size=self.n_ensembles,
                                            hidden_size=config.hidden_size,
                                            lr=config.mb_lr,
                                            device=device).to(device)
    

        self.n_rollouts = config.n_rollouts * config.parallel_envs
        self.rollout_select = config.rollout_select
        self.elite_size = config.elite_size
        self.elite_idxs = []
        
        self.max_not_improvements = 5
        self._current_best = [1e10 for i in range(self.n_ensembles)]
        self.improvement_threshold = 0.01
        self.break_counter = 0
        self.env_name = config.env
        
    def train(self, train_loader, test_loader):
        losses = 0
        epochs_trained = 0
        self.break_counter = 0
        break_training = False
        while True:
            self.dynamics_model.train()
            for (s, a, r, ns, d) in train_loader:
                delta_state = ns - s
                targets = torch.cat((delta_state, r), dim=-1).to(self.device)
                loss = self.dynamics_model.calc_loss(s, a, targets)
                self.dynamics_model.optimize(loss)
                epochs_trained += 1
                
            # evaluation
            self.dynamics_model.eval()
            with torch.no_grad():
                for (s, a, r, ns, d) in test_loader:
                    delta_state = ns - s
                    targets = torch.cat((delta_state, r), dim=-1).to(self.device)
                    val_losses = self.dynamics_model.calc_loss(s, a, targets, include_var=False)
                    val_losses = val_losses.detach().cpu().numpy()
                    sorted_loss_idx = np.argsort(losses)
                    self.elite_idxs = sorted_loss_idx[:self.elite_size].tolist()
                    break_training = self.test_break_condition(val_losses)
                    if break_training:
                        break
            if break_training:
                break

            
        assert len(val_losses) == self.n_ensembles, f"epoch_losses: {len(val_losses)} =/= {self.n_ensembles}"
        
        return val_losses, np.mean(epochs_trained)
    
    def test_break_condition(self, current_losses):
        keep_train = False
        for i in range(len(current_losses)):
            current_loss = current_losses[i]
            best_loss = self._current_best[i]
            improvement = (best_loss - current_loss) / best_loss
            if improvement > self.improvement_threshold:
                self._current_best[i] = current_loss
                keep_train = True
    
        if keep_train:
            self.break_counter = 0
        else:
            self.break_counter += 1
        if self.break_counter >= self.max_not_improvements:
            return True
        else:
            return False
            

    def run_ensemble_prediction(self, states, actions):
        with torch.no_grad():
            mus, var = self.dynamics_model(torch.from_numpy(states).float().to(self.device),
                                            torch.from_numpy(actions).float().to(self.device), 
                                            return_log_var=False)

        # [ensembles, batch, prediction_shape]
        assert mus.shape == (self.n_ensembles, states.shape[0], states.shape[1] + 1)
        assert var.shape == (self.n_ensembles, states.shape[0], states.shape[1] + 1)
        return mus.cpu().numpy(), var.cpu().numpy()


    def do_rollouts(self, buffer, env_buffer, policy, kstep):
        
        states, _, _, _, _ = env_buffer.sample(self.n_rollouts)
        states = states.cpu().numpy()
        steps_added = []
        for k in range(kstep):
            actions = policy.get_action(states)
            ensemble_means, ensemble_var = self.run_ensemble_prediction(states, actions)
            ensemble_means[:, :, :-1] += states
            ensemble_var = np.sqrt(ensemble_var)
            if self.probabilistic:
                all_ensemble_predictions = ensemble_means + np.random.normal(size=ensemble_means.shape) * ensemble_var
            else:
                all_ensemble_predictions = ensemble_means
            steps_added.append(len(states))
            if self.rollout_select == "random":
                # choose what predictions we select from what ensemble member
                ensemble_idx = random.choices(self.elite_idxs, k=self.n_rollouts)
                step_idx = np.arange(states.shape[0])
                # pick prediction based on ensemble idxs
                predictions = all_ensemble_predictions[ensemble_idx, step_idx, :]
            else:
                predictions = all_ensemble_predictions[self.elite_idxs].mean(0)
            assert predictions.shape == (self.n_rollouts, states.shape[1] + 1)

            next_states = predictions[:, :-1]
            rewards = predictions[:, -1]
            dones = termination_fn(self.env_name, states, actions, next_states, rewards)
            for (s, a, r, ns, d) in zip(states, actions, rewards, next_states, dones):
                buffer.add(s, a, r, ns, d)
            nonterm_mask = ~dones.squeeze(-1)
            if nonterm_mask.sum() == 0:
                break
            states = next_states[nonterm_mask]
        # calculate epistemic uncertainty ~ variance between the ensembles 
        # over the course of training ensembles should all predict the same variance -> 0 
        # model is very certain what will happen
        variance_over_each_state = all_ensemble_predictions.var(0)
        whole_batch_uncertainty = variance_over_each_state.var()
        # calculate mean rollout length
        mean_rollout_length = sum(steps_added) / self.n_rollouts
        return whole_batch_uncertainty.item(), mean_rollout_length

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
            
    
