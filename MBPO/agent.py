import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from networks import Critic, Actor
import copy


class SAC(nn.Module):
    """Interacts with and learns from the environment."""
    
    def __init__(self,
                        state_size,
                        action_size,
                        config,
                        device
                ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        super(SAC, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.device = device
        
        self.gamma = torch.FloatTensor([config.gamma]).to(device)
        self.tau = config.tau
        hidden_size = config.sac_hidden_size
        learning_rate = config.sac_lr
        self.clip_grad_param = config.clip_grad

        self.target_entropy = -action_size  # -dim(A)

        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate) 
                
        # Actor Network 

        self.actor_local = Actor(state_size, action_size, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)     
        
        # Critic Network (w/ Target Network)

        self.critic1 = Critic(state_size, action_size, hidden_size, 2).to(device)
        self.critic2 = Critic(state_size, action_size, hidden_size, 1).to(device)
        
        assert self.critic1.parameters() != self.critic2.parameters()
        
        self.critic1_target = Critic(state_size, action_size, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(state_size, action_size, hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate) 

    
    def get_action(self, state, eval=False):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        
        with torch.no_grad():
            if eval:
                action = self.actor_local.get_det_action(state)
            else:
                action = self.actor_local.get_action(state)
        return action.numpy()

    def calc_policy_loss(self, states, alpha):
        actions_pred, log_pis = self.actor_local.evaluate(states)

        q1 = self.critic1(states, actions_pred)   
        q2 = self.critic2(states, actions_pred)
        min_Q = torch.min(q1,q2)
        actor_loss = ((alpha * log_pis - min_Q )).mean()
        return actor_loss, log_pis
    
    def learn(self, fake_buffer, real_buffer, ratio):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        # batch_size = fake_buffer.batch_size
        # real_data_ratio = int(ratio * batch_size)
        # fake_data_ratio = batch_size - real_data_ratio
        #real_experience = real_buffer.sample(real_data_ratio)
        experience = fake_buffer.sample(256)
        
        #real_states, real_actions, real_rewards, real_next_states, real_dones = real_experience
        states, actions, rewards, next_states, dones = experience
        
        # states = torch.cat((real_states, fake_states))
        # actions = torch.cat((real_actions, fake_actions))
        # rewards = torch.cat((real_rewards, fake_rewards))
        # next_states = torch.cat((real_next_states, fake_next_states))
        # dones = torch.cat((real_dones, fake_dones))
        # ---------------------------- update actor ---------------------------- #
        current_alpha = copy.deepcopy(self.alpha)
        actor_loss, log_pis = self.calc_policy_loss(states, current_alpha.to(self.device))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Compute alpha loss
        alpha_loss = - (self.log_alpha.exp() * (log_pis + self.target_entropy).detach().cpu()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            next_action, log_pis_next = self.actor_local.evaluate(next_states)
            Q_target1_next = self.critic1_target(next_states, next_action)
            Q_target2_next = self.critic2_target(next_states, next_action)
            Q_target_next = torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * log_pis_next
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (self.gamma * (1 - dones) * Q_target_next) 

        # Compute critic loss
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
            
        critic1_loss = 0.5 * F.mse_loss(q1, Q_targets)
        critic2_loss = 0.5 * F.mse_loss(q2, Q_targets)

        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)
        
        return actor_loss.item(), alpha_loss.item(), critic1_loss.item(), critic2_loss.item(), current_alpha

    def soft_update(self, local_model , target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
