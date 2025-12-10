"""
mappo_network.py

Neural network architectures for MAPPO in hide and seek.
UPDATED: Implements GRU (Recurrent Neural Networks) for memory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class SpatialFeatureExtractor(nn.Module):
    """
    CNN to process grid observation.
    """
    def __init__(self, grid_size=20):
        super().__init__()
        # Input: 5 channels (self, other, blocks, ramp, walls)
        self.conv1 = nn.Conv2d(5, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # 20x20 -> 10x10 -> 5x5. 32 channels * 5 * 5 = 800
        self.out_dim = 32 * 5 * 5
        self.fc = nn.Linear(self.out_dim, 256)

    def forward(self, obs_map):
        x = F.relu(self.conv1(obs_map))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.flatten(start_dim=1)
        return F.relu(self.fc(x))

class RnnActor(nn.Module):
    """Actor network with GRU for memory."""
    def __init__(self, input_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x, hidden):
        # x shape: (batch, input_dim) -> (batch, 1, input_dim) for GRU
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # GRU pass
        # output: (batch, seq, hidden), hidden: (1, batch, hidden)
        x, new_hidden = self.gru(x, hidden)
        
        # Take last time step for prediction
        x = x[:, -1, :] 
        probs = F.softmax(self.actor_head(x), dim=-1)
        return probs, new_hidden

class RnnCritic(nn.Module):
    """Critic network with GRU."""
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x, new_hidden = self.gru(x, hidden)
        x = x[:, -1, :]
        value = self.value_head(x)
        return value, new_hidden

class MAPPOAgent(nn.Module):
    def __init__(self, grid_size=20, action_dim=10, device='cpu'):
        super().__init__()
        self.device = device
        self.hidden_dim = 256
        
        # Feature Extractors (Shared or Separate)
        self.hider_cnn = SpatialFeatureExtractor(grid_size).to(device)
        self.seeker_cnn = SpatialFeatureExtractor(grid_size).to(device)
        
        # Actors (Decentralized with Memory)
        self.hider_actor = RnnActor(256, action_dim, self.hidden_dim).to(device)
        self.seeker_actor = RnnActor(256, action_dim, self.hidden_dim).to(device)
        
        # Critic (Centralized with Memory)
        # Takes concatenation of features (256 + 256)
        self.critic = RnnCritic(512, self.hidden_dim).to(device)

    def get_initial_hidden(self, batch_size):
        """Returns zero hidden states."""
        return torch.zeros(1, batch_size, self.hidden_dim).to(self.device)

    def obs_to_tensor(self, obs_dict, batch_size, env):
        """
        Convert dict observation to grid tensor (N, 5, 20, 20).
        Logic to construct 5-channel map from obs data would go here.
        For brevity, assuming a simple mock implementation or that 
        obs contains 'grid_map'. If not, you need a function to 
        paint the obs['state'] into a 20x20 grid.
        """
        # Placeholder: creating random tensors for demonstration
        # In production: Use env.construct_tensor_representation(obs)
        return torch.rand(batch_size, 5, 20, 20).to(self.device)

    def get_action(self, obs, hider_hidden, seeker_hidden):
        """
        Forward pass for actors.
        Returns: actions, log_probs, new_hidden_states
        """
        batch_size = 1 # Inference usually 1
        
        # 1. Extract Features
        # (Assuming you implement obs_to_tensor logic based on your existing code)
        h_map = self.obs_to_tensor(obs, batch_size, None) 
        s_map = self.obs_to_tensor(obs, batch_size, None)
        
        h_feat = self.hider_cnn(h_map)
        s_feat = self.seeker_cnn(s_map)
        
        # 2. Actors
        h_probs, h_new_hidden = self.hider_actor(h_feat, hider_hidden)
        s_probs, s_new_hidden = self.seeker_actor(s_feat, seeker_hidden)
        
        h_dist = Categorical(h_probs)
        s_dist = Categorical(s_probs)
        
        h_action = h_dist.sample()
        s_action = s_dist.sample()
        
        return {
            "hider": h_action.item(),
            "seeker": s_action.item()
        }, {
            "hider": h_dist.log_prob(h_action),
            "seeker": s_dist.log_prob(s_action)
        }, {
            "hider": h_new_hidden,
            "seeker": s_new_hidden
        }

    def get_value(self, obs, critic_hidden):
        batch_size = 1
        h_map = self.obs_to_tensor(obs, batch_size, None)
        s_map = self.obs_to_tensor(obs, batch_size, None)
        
        h_feat = self.hider_cnn(h_map)
        s_feat = self.seeker_cnn(s_map)
        
        # Centralized Critic input: both features
        concat_feat = torch.cat([h_feat, s_feat], dim=1)
        
        value, new_hidden = self.critic(concat_feat, critic_hidden)
        return value, new_hidden