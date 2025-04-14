from PyLR3M.LR3M import LR3M

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

import os

def imagify(arr, maxval=None):
    if maxval == None:
        maxval = np.max(arr)
    arrs = np.clip(arr / maxval * 255, 0, 255)
    return arrs.astype('uint8')

enhancer = LR3M(alpha=0.015, beta=1.5e-3, eps=10, lmbda=2.5, sigma=10)

for f in os.listdir("data"):
    fbase = os.path.splitext(f)[0]
    im_in = np.array(Image.open(f'data/{f}'))

    Image.fromarray(im_in.astype('uint8')).save(f'base_{fbase}.png')

    Lhat, Gh, Gv = enhancer._initialize(im_in)
    L = enhancer._solve_L_subproblem(Lhat)
    Rhat = enhancer._initialize_Rhat(im_in,L)
    Rhat_kp1 = enhancer._solve_R_contrast_enhancement(Rhat, Gh, Gv)

    L3 = np.tile(pow(L[...,np.newaxis],1/2.2),(1,1,3))
    S_cen = Rhat_kp1 * L3
    S_base = Rhat * L3

    sat = 15
    
    stacked = np.concat((imagify(im_in),imagify(S_cen,sat)), axis=0)
    plt.imshow(stacked)
    plt.show()
