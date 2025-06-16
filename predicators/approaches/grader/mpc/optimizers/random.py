import numpy as np
import torch
import time

from .optimizer import Optimizer


class RandomOptimizer(Optimizer):
    def __init__(self, action_dim, horizon, popsize):
        super().__init__()
        self.horizon = horizon
        self.popsize = popsize
        self.action_dim = action_dim
        self.solution = None
        self.cost_function = None

    def setup(self, cost_function):
        self.cost_function = cost_function

    def reset(self):
        pass

    def generate_one_action(self, low, high, size):
        shape = torch.Size(size)
        if torch.cuda.is_available():
            move = torch.cuda.LongTensor(shape)
        else:
            move = torch.LongTensor(shape)

        torch.randint(0, high, size=shape, out=move)
        move = torch.nn.functional.one_hot(move)
        return move

    def obtain_solution_chemistry(self, action_dim):     
        # convert int to onehot
        action = np.random.randint(0, action_dim, size=(self.popsize, self.horizon))
        action = (np.arange(action_dim) == action[..., None]).astype(int)
        costs = self.cost_function(action)
        solution = action[np.argmin(costs)]
        return solution