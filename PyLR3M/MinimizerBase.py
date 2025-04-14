import numpy as np


class MinimizerBase:
    # implement __init__ function to take in parameters
    def __init__(self):
        pass

    # implement minimize function to take in Rbar, and do noise suppression
    def minimize(self,
                 im_in: np.ndarray
                 ):
        pass