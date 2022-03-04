import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import torch.nn.functional as F



def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size=32, init_w=3e-3, log_std_min=-20, log_std_max=2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std_linear = nn.Linear(hidden_size, action_size)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)

        return action, log_prob
        
    
    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        return action.detach().cpu()
    
    def get_det_action(self, state):
        mu, log_std = self.forward(state)
        return torch.tanh(mu).detach().cpu()


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_size=32, seed=1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in the network layers
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size+action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
def init_weights(m):
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        return t

    if type(m) == nn.Linear or isinstance(m, Ensemble_FC_Layer):
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.0)


class SimpleDynamics(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=200, lr=1e-2, device="cpu"):
        super(SimpleDynamics, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        self.model = nn.Sequential(nn.Linear(state_size + action_size, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, state_size + 1)).to(device)
        #self.apply(init_weights)
        self.loss_f = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        
    def forward(self, x):
        #assert x.device == self.model.device
        x = self.model(x)
        return x
    
    def calc_loss(self, x, target):
        assert x.shape[1] == self.state_size + self.action_size
        pred = self.forward(x)
        assert pred.shape == target.shape
        loss = self.loss_f(pred, target)
        return loss
    
    def optimize(self, x, target):
        self.optimizer.zero_grad()
        loss = self.calc_loss(x, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    

class Ensemble_FC_Layer(nn.Module):
    def __init__(self, in_features, out_features, ensemble_size, bias=True):
        super(Ensemble_FC_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        pass


    def forward(self, x) -> torch.Tensor:
        w_times_x = torch.bmm(x, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    
class DynamicsModel(nn.Module):

    def __init__(self, state_size, action_size, ensemble_size=7, hidden_size=200, lr=1e-2, device="cpu"):
        super(DynamicsModel, self).__init__()
        self.ensemble_size = ensemble_size
        self.output_size = state_size + 1
        self.fc1 = Ensemble_FC_Layer(state_size + action_size, hidden_size, ensemble_size)
        self.fc2 = Ensemble_FC_Layer(hidden_size, hidden_size, ensemble_size)
        self.fc3 = Ensemble_FC_Layer(hidden_size, hidden_size, ensemble_size)
        self.fc4 = Ensemble_FC_Layer(hidden_size, hidden_size, ensemble_size)
        self.output_layer = Ensemble_FC_Layer(hidden_size, self.output_size*2, ensemble_size)
        self.apply(init_weights)
        
        self.activation = nn.SiLU()

        self.min_logvar = nn.Parameter((-torch.ones((1, state_size + 1)).float() * 10).to(device), requires_grad=False)
        self.max_logvar = nn.Parameter((torch.ones((1, state_size + 1)).float() / 2).to(device), requires_grad=False)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
    def forward(self, x, return_log_var=False):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.fc4(x)
        x = self.activation(x)
        
        output = self.output_layer(x)
        mu = output[:, :, :self.output_size]
        log_var = output[:, :, self.output_size:]
        log_var = self.max_logvar - F.softplus(self.max_logvar - log_var)
        log_var = self.min_logvar + F.softplus(log_var - self.min_logvar)

        if return_log_var:
            return mu, log_var
        else:
            return mu, torch.exp(log_var)
    
    def calc_loss(self, inputs, targets, include_var=True):
        mu, log_var = self(inputs, return_log_var=True)
        if include_var:
            inv_var = (-log_var).exp()
            loss = ((mu - targets)**2 * inv_var).mean(-1).mean(-1).sum() + log_var.mean(-1).mean(-1).sum()
            return loss
        else:
            return ((mu - targets)**2).mean(-1).mean(-1)      

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        loss.backward()
        self.optimizer.step()