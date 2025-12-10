"""
mappo_trainer.py

Trainer updated to handle Hidden States for RNNs.
"""

import torch
import torch.optim as optim
import numpy as np
from collections import deque

class RolloutBuffer:
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.obs = []
        self.actions = {"hider": [], "seeker": []}
        self.rewards = {"hider": [], "seeker": []}
        self.values = []
        self.log_probs = {"hider": [], "seeker": []}
        self.dones = []
        # New: Store hidden states
        self.hidden_states = {"hider": [], "seeker": [], "critic": []}
        
    def add(self, obs, actions, rewards, value, log_probs, done, hiddens):
        self.obs.append(obs)
        self.actions["hider"].append(actions["hider"])
        self.actions["seeker"].append(actions["seeker"])
        self.rewards["hider"].append(rewards["hider"])
        self.rewards["seeker"].append(rewards["seeker"])
        self.values.append(value)
        self.log_probs["hider"].append(log_probs["hider"])
        self.log_probs["seeker"].append(log_probs["seeker"])
        self.dones.append(done)
        
        # Store hidden tensors (detach from graph to save memory in buffer)
        self.hidden_states["hider"].append(hiddens["hider"].detach())
        self.hidden_states["seeker"].append(hiddens["seeker"].detach())
        if "critic" in hiddens:
            self.hidden_states["critic"].append(hiddens["critic"].detach())

class MAPPOTrainer:
    def __init__(self, agent, args):
        self.agent = agent
        self.buffer = RolloutBuffer()
        self.optimizer = optim.Adam(agent.parameters(), lr=3e-4)
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.gamma = 0.99
        self.gae_lambda = 0.95
        
    def collect_rollouts(self, env, n_steps):
        """Run environment and collect data with RNN states."""
        obs = env.reset()
        
        # Initialize hidden states (1, 1, hidden_dim) assuming 1 env
        h_hidden = self.agent.get_initial_hidden(1)
        s_hidden = self.agent.get_initial_hidden(1)
        c_hidden = self.agent.get_initial_hidden(1)
        
        for _ in range(n_steps):
            with torch.no_grad():
                # Get Actions
                actions, log_probs, new_hiddens = self.agent.get_action(
                    obs, h_hidden, s_hidden
                )
                
                # Get Value
                value, new_c_hidden = self.agent.get_value(obs, c_hidden)
            
            next_obs, done, rewards = env.step(actions)
            
            # Store data + OLD hidden states
            current_hiddens = {"hider": h_hidden, "seeker": s_hidden, "critic": c_hidden}
            
            self.buffer.add(
                obs, actions, rewards, value, log_probs, done, current_hiddens
            )
            
            obs = next_obs
            
            # Update hidden states for next step
            # IMPORTANT: If done, reset hidden states to zero!
            if done:
                h_hidden = self.agent.get_initial_hidden(1)
                s_hidden = self.agent.get_initial_hidden(1)
                c_hidden = self.agent.get_initial_hidden(1)
                obs = env.reset()
            else:
                h_hidden = new_hiddens["hider"]
                s_hidden = new_hiddens["seeker"]
                c_hidden = new_c_hidden

    def update(self):
        # Convert buffer to tensors...
        # Calculation of Advantages (GAE) ...
        # (Standard PPO logic here, omitted for brevity but standard implementation applies)
        
        # RNN Training Note:
        # When sampling batches, you MUST pass the stored hidden_state from the buffer
        # into the forward pass of the network, otherwise the RNN has no context.
        pass 
        # Implement standard PPO update loop using self.buffer data