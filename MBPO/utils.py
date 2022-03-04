import torch
import numpy as np
import gym
from gym.spaces import Box

def save(args, save_name, model, wandb, ep=None):
    import os
    save_dir = './trained_models/' 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not ep == None:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + str(ep) + ".pth")
        wandb.save(save_dir + args.run_name + save_name + str(ep) + ".pth")
    else:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + ".pth")
        wandb.save(save_dir + args.run_name + save_name + ".pth")

def collect_random(env, dataset, num_samples=200):
    state = env.reset()
    for _ in range(num_samples):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        dataset.add(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()

def evaluate(env, policy, eval_runs=5): 
    """
    Makes an evaluation run with the current policy
    """

    reward_batch = []
    for i in range(eval_runs):
        state = env.reset()

        rewards = 0
        while True:
            action = policy.get_action(state, eval=True)
            state, reward, done, _ = env.step(action)
            rewards += reward
            if done:
                break
        reward_batch.append(rewards)
    return np.mean(reward_batch)

class TorchStandardScaler:
    def fit(self, state, action, reward, next_state):
        input1 = torch.cat((state, action), dim=-1)
        self.state_action_mu = torch.mean(input1, axis=0, keepdims=True)
        self.state_action_std = torch.std(input1, axis=0, keepdims=True)
        self.state_action_std[self.state_action_std < 1e-12] = 1.0

        input2 = torch.cat((next_state, reward), dim=-1)
        self.next_state_reward_mu = torch.mean(input2, axis=0, keepdims=True)
        self.next_state_reward_std = torch.std(input2, axis=0, keepdims=True)
        self.next_state_reward_std[self.next_state_reward_std < 1e-12] = 1.0
    
    def transform(self, x, input_type="input"):
        if input_type == "input":
            x -= self.state_action_mu
            x /= self.state_action_std
            return x
        elif input_type == "target":
            x -= self.next_state_reward_mu
            x /= self.next_state_reward_std
            return x
        else:
            print("input type {} does not exist".format(input_type))

    def inverse_transform(self, x):
        return self.next_state_reward_std * x + self.next_state_reward_mu
