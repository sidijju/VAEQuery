import gym
import random
from gym import spaces
import numpy as np
import torch

class GridWorld(gym.Env):

    def __init__(self, args):

        self.args = args

        self.horizon = args.max_trajectory_len
        assert self.horizon > 0
       
        self.size = args.grid_size
        assert self.size > 1

        # for gridworld, we cannot observe target
        self.observation_space = spaces.Box(low=0, high=self.size-1, shape=(2,))
        self.action_space = spaces.Discrete(5)

         # list of possible tasks (exclude tasks that are trivial)
        self.possible_tasks = [(i, j) for i in range(self.size) for j in range(self.size) if (i > 1 or j > 1)]

        self.task_dim = 2
        self.state_dim = 2
        self.action_dim = 1

        # set initial vars
        self.reset()
    
    def reset(self):
        self.step_count = 0
        self.state = np.array((0.0, 0.0))
        self.task = np.array(random.choice(self.possible_tasks))
        self.done = False
        return torch.tensor(self.state).to(self.args.device)

    def transition(self, action):
        # R, L, U, D, S
        assert action >= 0
        assert action <= 4

        if action == 0:
            self.state[0] = min(self.state[0] + 1, self.size - 1)
        elif action == 1:
            self.state[0] = max(self.state[0] - 1, 0)
        elif action == 2:
            self.state[1] = min(self.state[1] + 1, self.size - 1)
        elif action == 3:
            self.state[1] = max(self.state[1] - 1, 0)

        self.step_count += 1

        return self.state

    def step(self, action):

        if self.step_count <= self.horizon:
            self.transition(action)  
            self.step_count += 1

        done = self.step_count > self.horizon

         # the reward function is randomly chosen based off the task chosen
        if tuple(self.state) == tuple(self.task):
            reward = 1.0
        else:
            reward = -0.1

        info = {"task": self.task,
                "step_count": self.step_count}

        return torch.tensor(self.state).to(self.args.device), \
               torch.tensor(reward).to(self.args.device), \
               torch.tensor(done).to(self.args.device), info