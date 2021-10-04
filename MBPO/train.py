

import gym
import pybullet_envs
import numpy as np
from collections import deque
import torch
import wandb
import argparse
from buffer import ReplayBuffer, MBReplayBuffer
import glob
from utils import save, collect_random
import random
from agent import SAC
from model import MBEnsemble

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="MBPO-SAC", help="Run name, default: MBPO-SAC")
    parser.add_argument("--env", type=str, default="Pendulum-v0", help="Gym environment name, default: Pendulum-v0")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes, default: 100")
    parser.add_argument("--episode_length", type=int, default=1000, help="Length of one episode, default: 1000")
    parser.add_argument("--buffer_size", type=int, default=250_000, help="Maximal training dataset size, default: 250_000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--log_video", type=int, default=0, help="Log agent behaviour to wanbd when set to 1, default: 0")
    parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size, default: 256")
    parser.add_argument("--npolicy_updates", type=int, default=20, help="")
    
    # SAC params
    parser.add_argument("--gamma", type=float, default=0.99, help="")
    parser.add_argument("--tau", type=float, default=5e-3, help="")
    parser.add_argument("--sac_hidden_size", type=int, default=256, help="")
    parser.add_argument("--sac_lr", type=float, default=5e-4, help="")
    parser.add_argument("--clip_grad", type=float, default=10, help="")
    ## MB params
    parser.add_argument("--n_updates", type=int, default=5, help="")
    parser.add_argument("--mb_buffer_size", type=int, default=100_000, help="")
    parser.add_argument("--n_rollouts", type=int, default=400, help="")
    parser.add_argument("--ensembles", type=int, default=7, help="")
    parser.add_argument("--hidden_size", type=int, default=200, help="")
    parser.add_argument("--mb_lr", type=float, default=3e-4, help="")
    parser.add_argument("--mve_horizon", type=int, default=1, help="Model Based Value Expansion Horizon, default: 1")
    parser.add_argument("--rollout_select", type=str, default="random", choices=["random", "mean"], help="Define how the rollouts are composed, randomly from a random selected member of the ensemble or as the mean over all ensembles, default: random")
    
    # kstep schedule
    parser.add_argument("--kstep_start", type=int, default=1, help="kstep starting value")
    parser.add_argument("--kstep_end", type=int, default=1, help="kstep ending value")
    parser.add_argument("--epis_start", type=int, default=1, help="starting episode when the kstep value should be adapted")
    parser.add_argument("--epis_end", type=int, default=2, help="ending episode when the kstep value should have the 'kstep_end' value")
    
    args = parser.parse_args()
    return args 

def get_kstep(e, kstep_start, kstep_end, epis_start, epis_end):
    return int(min(max(kstep_start + ((e-epis_start)/(epis_end-epis_start)) * (kstep_end-kstep_start), kstep_start), kstep_end))

def train(config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    env = gym.make(config.env)
    
    env.seed(config.seed)
    env.action_space.seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    steps = 0
    average10 = deque(maxlen=10)
    total_steps = 0
    
    with wandb.init(project="MBPO", name=config.run_name, config=config):
        
        agent = SAC(state_size=env.observation_space.shape[0],
                    action_size=env.action_space.shape[0],
                    config=config,
                    device=device)
        
        ensemble = MBEnsemble(state_size=env.observation_space.shape[0],
                              action_size=env.action_space.shape[0],
                              config=config,
                              device=device)    
        
        wandb.watch(agent, log="gradients", log_freq=10)

        buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=config.batch_size, device=device)
        
        mb_buffer = MBReplayBuffer(buffer_size=config.mb_buffer_size,
                                   batch_size=config.n_rollouts,
                                   device=device)

        collect_random(env=env, dataset=mb_buffer, num_samples=5000)
        if config.log_video:
            env = gym.wrappers.Monitor(env, './video', video_callable=lambda x: x%10==0, force=True)
        for i in range(1, config.episodes+1):
            loss, reward_diff  = ensemble.train(mb_buffer.get_dataloader(batch_size=32))
            wandb.log({"Episode": i, "MB Loss": loss, "Reward-diff": reward_diff}, step=steps)
            state = env.reset()
            episode_steps = 0
            rewards = 0
            total_rewards = []
            epistemic_uncertainty_ = []
            for _ in range(config.episode_length):
                action = agent.get_action(state)
                steps += 1
                next_state, reward, done, _ = env.step(action)
                mb_buffer.add(state, action, reward, next_state, done)

                kstep = get_kstep(e=i, kstep_start=config.kstep_start,
                                  kstep_end=config.kstep_end,
                                  epis_start=config.epis_start,
                                  epis_end=config.epis_end)
                
                epistemic_uncertainty = ensemble.do_rollouts(buffer=buffer, env_buffer=mb_buffer, policy=agent, kstep=kstep)
                epistemic_uncertainty_.append(epistemic_uncertainty)
                for _ in range(config.npolicy_updates):
                    policy_loss, alpha_loss, bellmann_error1, bellmann_error2, current_alpha = agent.learn(buffer.sample(), ensemble)
                state = next_state
                rewards += reward
                episode_steps += 1
                if done:
                    state = env.reset()
                    total_rewards.append(rewards)
                    rewards = 0

            average10.append(np.mean(total_rewards))
            total_steps += episode_steps
            print("Episode: {} | Reward: {} | Polciy Loss: {} | Steps: {}".format(i, np.mean(total_rewards), policy_loss, steps,))
            
            wandb.log({"Reward": np.mean(total_rewards),
                       "Average10": np.mean(average10),
                       "Steps": total_steps,
                       "Policy Loss": policy_loss,
                       "Alpha Loss": alpha_loss,
                       "Bellmann error 1": bellmann_error1,
                       "Bellmann error 2": bellmann_error2,
                       "Alpha": current_alpha,
                       "Epistemic uncertainty": np.mean(epistemic_uncertainty_),
                       "Steps": steps,
                       "Kstep": kstep,
                       "Episode": i,
                       "Buffer size": buffer.__len__(),
                       "Env Buffer size": mb_buffer.__len__()})

            if (i %10 == 0) and config.log_video:
                mp4list = glob.glob('video/*.mp4')
                if len(mp4list) > 1:
                    mp4 = mp4list[-2]
                    wandb.log({"gameplays": wandb.Video(mp4, caption='episode: '+str(i-10), fps=4, format="gif"), "Episode": i})

            if i % config.save_every == 0:
                save(config, save_name="MBPO-SAC", model=agent.actor_local, wandb=wandb, ep=0)

if __name__ == "__main__":
    config = get_config()
    train(config)
