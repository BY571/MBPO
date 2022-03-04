from networks import DynamicsModel, SimpleDynamics
import torch
import random
import numpy as np
from utils import TorchStandardScaler
from torch.utils.data import TensorDataset, DataLoader


def get_termination_fn(env_name):
    if env_name == "Hopper-v2":
        def termination_fn(obs, act, next_obs, rewards):
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
        return termination_fn
    elif env_name == "Walker2dBulletEnv-v0":
        def termination_fn(obs, act, next_obs, rewards):
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
        return termination_fn
    else:
        def termination_fn(obs, act, next_obs, rewards):
            done = torch.zeros(obs.shape[0]).bool()
            return done
        return termination_fn


class DynamicsModel():
    def __init__(self, env_name, state_size, action_size, n_trajectories, num_epochs=60, batch_size=512, hidden_size=200, lr=1e-2, device="cpu"):
        
        self.device = device
        self.n_trajectories = n_trajectories
        self.num_epochs = num_epochs
        self.batch_size = 512
        self.dynamics_model = SimpleDynamics(state_size=state_size, action_size=action_size,
                                             hidden_size=hidden_size, lr=lr, device=device)
        
        self.terminate_function = get_termination_fn(env_name=env_name)
        self.scaler = TorchStandardScaler()
        
    def fit(self, dynamics_buffer):
        
        states, actions, rewards, next_states = dynamics_buffer.sample()
        # print(states.shape, actions.shape, rewards.shape, next_states.shape)
        self.scaler.fit(states, actions, rewards, next_states)
        inputs_v = self.process_dynamics_input(input1=states, input2=actions, input_type="input")
        targets_v = self.process_dynamics_input(input1=next_states, input2=rewards, input_type="target")
        dataset = TensorDataset(inputs_v, targets_v)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        print("\n -- Dynamics - Training -- ")
        episode_losses = []
        for ep in range(1, self.num_epochs+1):
            batch_losses = []
            for (x, y) in data_loader:
                loss = self.dynamics_model.optimize(x, y)
                batch_losses.append(loss)
            #print("Episode: {} | Loss: {}".format(ep, np.mean(batch_losses)))
            episode_losses.append(np.mean(batch_losses))
        return np.mean(episode_losses)

    def process_dynamics_input(self, input1, input2, input_type="input"):
        values = torch.cat((input1, input2), dim=-1)
        norm_values = self.scaler.transform(values, input_type).to(self.device)
        return norm_values
    
    def inv_transform_predictions(self, prediction):
        denormalized_prediction = self.scaler.inverse_transform(prediction)
        states = denormalized_prediction[:, :-1]
        rewards = denormalized_prediction[:, -1]
        return states, rewards

    def rollout_prediction(self, agent_buffer, dynamics_buffer, policy, rollout_horizon):
        # sample random starting point for the rollouts
        states = dynamics_buffer.sample_random_state(self.n_trajectories) # shape [number of trajectories, state size]
        # to count how many transition tuples get added
        steps_added = []
        for i in range(rollout_horizon):
            steps_added.append(states.shape[0])
            # policy predict actions to take in state
            actions = policy.get_action(states.cpu().detach().numpy())
            actions = torch.from_numpy(actions).float().to(self.device)
            norm_inputs = self.process_dynamics_input(states, actions)
            predictions = self.dynamics_model(norm_inputs)
            next_states, rewards = self.inv_transform_predictions(predictions)
            dones = self.terminate_function(states.cpu().detach().numpy(), actions.cpu().detach().numpy(), next_states.cpu().detach().numpy(), rewards.cpu().detach().numpy())
            for (s, a, r, ns, d) in zip(states.cpu().detach().numpy(), actions.cpu().detach().numpy(), rewards.cpu().detach().numpy(), next_states.cpu().detach().numpy(), dones):
                agent_buffer.add(s, a, r, ns, d)
            nonterm_mask = ~dones.squeeze(-1)
            if nonterm_mask.sum() == 0:
                break
            states = next_states[nonterm_mask].detach()

        # calculate mean rollout length
        mean_rollout_length = sum(steps_added) / self.n_trajectories
        return mean_rollout_length
