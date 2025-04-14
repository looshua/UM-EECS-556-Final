from PyLR3M.LR3M import LR3M

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

import os

'''
This example code goes through every file in the data folder and brightens it using the
selected minimizer.
'''

def imagify(arr, maxval=None):
    if maxval == None:
        maxval = np.max(arr)
    arrs = np.clip(arr / maxval * 255, 0, 255)
    return arrs.astype('uint8')

# change parameters as needed
enhancer = LR3M(alpha=0.015, beta=1.5e-3, eps=10, lmbda=2.5, sigma=10)

# SET MINIMIZER 
minimizer = None

enhancer.set_minimizer(minimizer)

for f in os.listdir("data"):
    fbase = os.path.splitext(f)[0]
    im_in = np.array(Image.open(f'data/{f}'))

    Image.fromarray(im_in.astype('uint8')).save(f'base_{fbase}.png')

    Lhat, Gh, Gv = enhancer._initialize(im_in)
    L = enhancer._solve_L_subproblem(Lhat)
    Rhat = enhancer._initialize_Rhat(im_in,L)
    Rhat_kp1 = enhancer._solve_R_contrast_enhancement(Rhat, Gh, Gv)
    Rkp1 = enhancer._solve_R_noise_suppression(im_in,Rhat_kp1,L)

    S_improve = enhancer.brighten(L,Rkp1,2.2)

    # change this to control the maximum brightness of the adjusted image
    SAT = 15
    
    stacked = np.concat((imagify(im_in),imagify(S_improve,SAT)), axis=0)
    # Image.fromarray(stacked).save(f'enhanced_{fbase}.png')
    plt.imshow(stacked)
    plt.show()
