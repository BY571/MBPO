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
from utils import evaluate, TorchStandardScaler
import multipro
from tqdm import tqdm

def get_config():
    parser = argparse.ArgumentParser(   )
    parser.add_argument("--run_name", type=str, default="MBPO-combined-mu-var-layer", help="Run name, default: MBPO-SAC")
    parser.add_argument("--env", type=str, default="Pendulum-v0", help="Gym environment name, default: Pendulum-v0")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes, default: 100")
    parser.add_argument("--episode_length", type=int, default=1000, help="Length of one episode, default: 1000")
    parser.add_argument("--buffer_size", type=int, default=1_000_000, help="Maximal training dataset size, default: 1_000_000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--log_video", type=int, default=0, help="Log agent behaviour to wanbd when set to 1, default: 0")
    parser.add_argument("--save_every", type=int, default=5, help="Saves the network every x epochs, default: 5")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size, default: 256")
    parser.add_argument("--npolicy_updates", type=int, default=20, help="")
    parser.add_argument("--max_train_repeat_per_timestep", type=int, default=5, help="")
    parser.add_argument("--parallel_envs", type=int, default=1, help="")
    
    # SAC params
    parser.add_argument("--gamma", type=float, default=0.99, help="")
    parser.add_argument("--tau", type=float, default=5e-3, help="")
    parser.add_argument("--sac_hidden_size", type=int, default=256, help="")
    parser.add_argument("--sac_lr", type=float, default=5e-4, help="")
    parser.add_argument("--clip_grad", type=float, default=10, help="")
    parser.add_argument("--real_data_ratio", type=float, default=0.1, help="")
    ## MB params
    parser.add_argument("--model_based_batch_size", type=int, default=256, help="")
    parser.add_argument("--n_rollouts", type=int, default=100_000, help="")
    parser.add_argument("--ensembles", type=int, default=7, help="")
    parser.add_argument("--elite_size", type=int, default=5, help="")
    parser.add_argument("--hidden_size", type=int, default=200, help="")
    parser.add_argument("--mb_lr", type=float, default=1e-3, help="")
    parser.add_argument("--update_frequency", type=int, default=250, help="")
    parser.add_argument("--rollout_select", type=str, default="random", choices=["random", "mean"], help="Define how the rollouts are composed, randomly from a random selected member of the ensemble or as the mean over all ensembles, default: random")
    
    # kstep schedule
    parser.add_argument("--kstep_start", type=int, default=1, help="kstep starting value")
    parser.add_argument("--kstep_end", type=int, default=1, help="kstep ending value")
    parser.add_argument("--epis_start", type=int, default=1, help="starting episode when the kstep value should be adapted")
    parser.add_argument("--epis_end", type=int, default=2, help="ending episode when the kstep value should have the 'kstep_end' value")
    
    args = parser.parse_args()
    return args 

def get_kstep(e, kstep_start, kstep_end, epis_start, epis_end):
    return int(min(max(kstep_start + (e-epis_start)/(epis_end-epis_start) * (kstep_end-kstep_start), kstep_start), kstep_end))

def calc_mb_buffer_size(config, rollout_length):
    rollouts_per_epoch = config.n_rollouts * config.episode_length / config.update_frequency
    model_steps_per_epoch = int(rollout_length * rollouts_per_epoch)
    return model_steps_per_epoch

def resize_buffer(config, kstep, buffer, device):
    buffer_size = calc_mb_buffer_size(config, kstep)
    all_samples = buffer.return_all()
    mb_buffer = MBReplayBuffer(buffer_size=buffer_size,
                                device=device)
    mb_buffer.push_batch(all_samples)
    return buffer
    
    
def train(config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    envs = multipro.SubprocVecEnv([lambda: gym.make(config.env) for i in range(config.parallel_envs)])
    evaluation_env = gym.make(config.env)
    envs.seed(config.seed)
    evaluation_env.seed(config.seed+1234)
    
    state_size = evaluation_env.observation_space.shape[0]
    action_size = evaluation_env.action_space.shape[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    steps = 0
    total_policy_updates = 0
    average10 = deque(maxlen=10)
    kstep = 1    
    with wandb.init(project="MBPO-tests", name=config.run_name, config=config):
        
        agent = SAC(state_size=state_size,
                    action_size=action_size,
                    config=config,
                    device=device)
        
        ensemble = MBEnsemble(state_size=state_size,
                              action_size=action_size,
                              config=config,
                              device=device)
        
        wandb.watch(agent, log="gradients", log_freq=10)

        buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=config.batch_size, device=device)
        
        buffer_size = calc_mb_buffer_size(config, 1)
        mb_buffer = MBReplayBuffer(buffer_size=buffer_size,
                                   device=device)

        collect_random(env=evaluation_env, dataset=mb_buffer, num_samples=5000)
        if config.log_video:
            evaluation_env = gym.wrappers.Monitor(evaluation_env, './video', video_callable=lambda x: x%10==0, force=True)

        # do training
        for i in tqdm(range(1, config.episodes+1)):
            state = envs.reset()
            episode_steps = 0
            episode_trainigs = 0
            while episode_steps < config.episode_length:

                if steps > 0 and steps % config.update_frequency == 0:
                    train_inputs, train_labels = mb_buffer.get_dataloader(batch_size=config.model_based_batch_size)
                    losses, trained_epochs = ensemble.train(train_inputs, train_labels)

                    new_kstep = get_kstep(e=i, kstep_start=config.kstep_start,
                                    kstep_end=config.kstep_end,
                                    epis_start=config.epis_start,
                                    epis_end=config.epis_end)

                    if kstep != new_kstep:
                        kstep = new_kstep
                    mb_buffer = resize_buffer(config, kstep, mb_buffer, device)
                    epistemic_uncertainty, mean_rollout_length = ensemble.do_rollouts(buffer=buffer,
                                                                                      env_buffer=mb_buffer,
                                                                                      policy=agent,
                                                                                      kstep=kstep)
                    tqdm.write("\nEpisode: {} | Ensemble losses: {}".format(i, losses))
                    wandb.log({"Episode": i,
                               "MB mean loss": np.mean(losses),
                               "MB mean trained epochs": trained_epochs,
                               "Epistemic Uncertainty": epistemic_uncertainty,
                               "Mean rollout length": mean_rollout_length,
                               "Kstep": kstep,}, step=steps)
                           
                # Interact with the real environment
                action = agent.get_action(state)
                next_state, reward, done, _ = envs.step(action)
                mb_buffer.add(state, action, reward, next_state, done)
                state = next_state
                
                # Train Policy 
                if episode_trainigs < config.max_train_repeat_per_timestep * steps:
                    for _ in range(config.npolicy_updates * config.parallel_envs):
                        policy_loss, alpha_loss, bellmann_error1, bellmann_error2, current_alpha = agent.learn(buffer,
                                                                                                            mb_buffer,
                                                                                                            config.real_data_ratio)
                        episode_trainigs += config.npolicy_updates * config.parallel_envs
                    wandb.log({"Policy Loss": policy_loss,
                               "Alpha Loss": alpha_loss,
                               "Bellman error 1": bellmann_error1,
                               "Bellman error 2": bellmann_error2,
                               "Alpha": current_alpha}, step=steps)
                    
                episode_steps += config.parallel_envs
                steps += config.parallel_envs
                if done.any():
                    state = envs.reset()
                    
            total_policy_updates += episode_trainigs
            # do evaluation runs 
            rewards = evaluate(evaluation_env, agent)
            
            average10.append(rewards)
            tqdm.write("\nEpisode: {} | Reward: {} | Polciy Loss: {} | Steps: {}".format(i, rewards, policy_loss, steps,))
            
            wandb.log({"Reward": rewards,
                       "Average10": np.mean(average10),
                       "Total Policy Updates": total_policy_updates,
                       "Steps": steps,
                       "Episode": i,
                       "Buffer size": buffer.__len__(),
                       "Env Buffer size": mb_buffer.__len__()})

            # log evaluation runs to wandb
            if config.log_video:
                mp4list = glob.glob('video/*.mp4')
                if len(mp4list) > 1:
                    mp4 = mp4list[-2]
                    wandb.log({"gameplays": wandb.Video(mp4, caption='episode: '+str(i-10), fps=4, format="gif"), "Episode": i})

            if i % config.save_every == 0:
                save(config, save_name="MBPO-SAC", model=agent.actor_local, wandb=wandb, ep=0)

if __name__ == "__main__":
    config = get_config()
    train(config)
