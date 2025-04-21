
import numpy as np
from PyLR3M.MinimizerBase import MinimizerBase
from PyLR3M.FNLM import FNLMMinimizer
from PyLR3M.BM3D import BM3DMinimizer

class FNLM_BM3D_Minimizer(MinimizerBase):
    def __init__(self, fnlm_params=None, bm3d_sigma=4):
        super().__init__()
        self.fnlm = FNLMMinimizer(**(fnlm_params or {}))
        self.bm3d = BM3DMinimizer(noise_sigma=bm3d_sigma)

    def minimize(self, im_in: np.ndarray):
        # Step 1: Apply FNLM to remove light noise and preserve texture
        R_fnlm = self.fnlm.minimize(im_in)
        # Step 2: Apply BM3D to refine structure and suppress residuals
        R_final = self.bm3d.minimize(R_fnlm)
        return R_final
