import torch
import numpy as np

class planar2d:
    """
    Class for planar 2D robot
    """
    def __init__(self, T,):
        self.T = T
        self.x_dim = 3
        self.z_dim = 2

    def trn(self):
        # transition probability
        return 0

    def obs(self):
        # observation probability
        return 0

    def log_prob(self):
        return np.log(1.0)

    def generate_data(self,T):
        x = torch.zeros(T,self.x_dim)
        z = torch.zeros(T,self.z_dim)

        for t in range(T):
            x[t,:] = 1.0
            z[t,:] = 1.0
        return x, z