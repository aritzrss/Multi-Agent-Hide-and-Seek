"""
env/hide_and_seek_env.py

UPDATED VERSION with:
- Random Room Placement (Generalization)
- Dense Rewards (Reward Shaping for faster learning)
- Height system & Mechanics preserved
"""

import gym
from gym import spaces
import numpy as np
import random
from env.room import Room
from env.objects import Block, Ramp
from agents.seeker import Seeker
from agents.hider import Hider
from utils.logger import log_debug, log_info

class HideAndSeekEnv(gym.Env):
    def __init__(self):
        super(HideAndSeekEnv, self).__init__()

        self.grid_size = 20
        self.room_size = 8
        
        # Action space (10 actions)
        # 0-3: Move, 4-7: Grab+Move, 8: Climb, 9: Lock
        self.action_space = spaces.Discrete(10)

        # Observation space adapted for the network
        self.observation_space = spaces.Dict({
            "seeker": spaces.Box(low=0, high=self.grid_size, shape=(4,), dtype=np.float32),
            "hider": spaces.Box(low=0, high=self.grid_size, shape=(4,), dtype=np.float32)
        })

        self.max_steps = 100
        self.current_step = 0

    def _initialize_objects(self):
        """Randomly place objects inside the room."""
        self.blocks = []
        
        # Helper to get random pos inside room
        def get_random_room_pos():
            rx = random.randint(self.room.top_left[0] + 1, self.room.top_left[0] + self.room.width - 2)
            ry = random.randint(self.room.top_left[1] + 1, self.room.top_left[1] + self.room.height - 2)
            return (rx, ry)

        # Create 2 blocks
        for _ in range(2):
            pos = get_random_room_pos()
            # Avoid placing on top of each other
            while any(b.position == pos for b in self.blocks):
                pos = get_random_room_pos()
            self.blocks.append(Block(pos))
            
        # Create 1 ramp outside or inside (randomly)
        if random.random() < 0.5:
            # Inside
            r_pos = get_random_room_pos()
            while any(b.position == r_pos for b in self.blocks):
                r_pos = get_random_room_pos()
        else:
            # Random position outside room
            r_pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            
        self.ramp = Ramp(r_pos)

    def reset(self):
        """Resets the env with RANDOM room position."""
        self.current_step = 0
        self.visibility_count = 0
        
        # 1. Randomize Room Position
        # Room is 8x8. Max top-left index is 20 - 8 = 12
        rx = random.randint(0, 12)
        ry = random.randint(0, 12)
        self.room = Room(top_left=(rx, ry), width=self.room_size, height=self.room_size)
        
        # 2. Initialize Objects based on new room
        self._initialize_objects()
        
        # 3. Initialize Agents
        # Hider starts inside room
        hider_x = random.randint(self.room.top_left[0] + 1, self.room.top_left[0] + self.room.width - 2)
        hider_y = random.randint(self.room.top_left[1] + 1, self.room.top_left[1] + self.room.height - 2)
        self.hider = Hider((hider_x, hider_y))
        
        # Seeker starts far away (random corner not in room)
        corners = [(0,0), (0, 19), (19, 0), (19, 19)]
        # Pick a corner far from the room center
        best_corner = max(corners, key=lambda c: abs(c[0]-rx) + abs(c[1]-ry))
        self.seeker = Seeker(best_corner)
        
        self.seeker_active = False # Preparation phase first
        self.prep_timer = 0
        self.prep_phase_duration = 15 # Give hider 15 steps to prepare

        # For dense rewards
        self.prev_distance = self._get_distance()

        return self._get_obs()

    def _get_distance(self):
        """Euclidean distance between agents."""
        sx, sy = self.seeker.position
        hx, hy = self.hider.position
        return np.sqrt((sx-hx)**2 + (sy-hy)**2)

    def compute_visible_cells(self, agent_state):
        # Placeholder for raycasting visibility logic
        # In a full implementation, use raycasting here.
        # For now, returning radius around agent
        x, y, _, _ = agent_state
        visible = []
        r = 5
        for i in range(x-r, x+r+1):
            for j in range(y-r, y+r+1):
                if 0 <= i < self.grid_size and 0 <= j < self.grid_size:
                    visible.append((i, j))
        return visible

    def step(self, actions):
        rewards = {"hider": 0.0, "seeker": 0.0}
        self.current_step += 1
        
        # PREPARATION PHASE LOGIC
        if self.prep_timer < self.prep_phase_duration:
            self.prep_timer += 1
            # Seeker is frozen
            actions["seeker"] = -1 # No action
        else:
            self.seeker_active = True

        # 1. Execute Actions
        # (Assuming agents implement .move() and interactions logic)
        # Simplify step call for brevity, assumes standard logic exists
        self.hider.step(actions["hider"], self.room, self.blocks, self.ramp, self.grid_size)
        if self.seeker_active:
            self.seeker.step(actions["seeker"], self.room, self.blocks, self.ramp, self.grid_size)

        # 2. Check Visibility (Line of Sight)
        # Simple proximity check for this snippet
        distance = self._get_distance()
        is_visible = distance < 5 # Simplified visibility
        
        if self.seeker_active:
            if is_visible:
                self.visibility_count += 1
                rewards["seeker"] += 0.5  # Reward for seeing
                rewards["hider"] -= 0.5   # Penalty for being seen
            else:
                rewards["hider"] += 0.1   # Reward for hiding successfully
                rewards["seeker"] -= 0.1  # Time penalty
            
            # 3. DENSE REWARD (Shaping)
            # Reward seeker for getting closer, penalize for moving away
            dist_delta = self.prev_distance - distance
            rewards["seeker"] += dist_delta * 1.0 
            self.prev_distance = distance

        # 4. Check Done
        done = self.current_step >= self.max_steps
        
        # Terminal Rewards
        if done:
            vis_ratio = self.visibility_count / max(1, (self.max_steps - self.prep_phase_duration))
            if vis_ratio > 0.4: # Win condition for Seeker
                rewards["seeker"] += 10.0
                rewards["hider"] -= 10.0
            else:
                rewards["hider"] += 10.0
                rewards["seeker"] -= 10.0

        return self._get_obs(), done, rewards

    def _get_obs(self):
        return {
            "seeker": {
                "state": self.seeker.get_state() if self.seeker_active else (-1, -1, -1, -1),
                "visible": self.compute_visible_cells(self.seeker.get_state()) if self.seeker_active else []
            },
            "hider": {
                "state": self.hider.get_state(),
                "visible": self.compute_visible_cells(self.hider.get_state())
            }
        }