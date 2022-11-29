import numpy as np

class Policy:
    def __init__(self):
        self.vis_directory = "visualizations/"

    def run_policy(self, query):
        pass

    def train_policy(self, n=1000):
        pass

class RandomPolicy(Policy):
    def __init__(self):
        super().__init__()
        self.vis_directory += "random/"

    def run_policy(self, query):
        return np.random.randint(high=len(query))

class GreedyPolicy(Policy):
    def __init__(self):
        super().__init__()
        self.vis_directory += "greedy/"

    def run_policy(self, query):
        # TODO
        pass

class RLPolicy(Policy):
    def __init__(self):
        super().__init__()
        self.vis_directory += "rl/"
    
    def run_policy(self, query):
        # TODO
        pass

    def train_policy(self, n=1000):
        # TODO
        pass