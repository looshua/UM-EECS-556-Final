import bm3d
from numpy import ndarray

from PyLR3M.MinimizerBase import MinimizerBase

class BM3DMinimizer(MinimizerBase):
    def __init__(self,noise_sigma=10):
        self.noise_sigma = noise_sigma
        self.k = 0

    def minimize(self, im_in: ndarray):
        self.k += 1
        self.noise_sigma = self.noise_sigma / self.k
        return bm3d.bm3d(im_in,[self.noise_sigma, 
                                self.noise_sigma,  
                                self.noise_sigma])