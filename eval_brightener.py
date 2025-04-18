from PyLR3M.LR3M import LR3M
from PyLR3M.BM3D import BM3DMinimizer

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import os

def imagify(arr, maxval=None):
    if maxval == None:
        maxval = np.max(arr)
    arrs = np.clip(arr / maxval * 255, 0, 255)
    return arrs.astype('uint8')

IMAGE_DIR = '/Users/looshua/masters/winter/ece 556/final/LOLdataset/our485/low'
SAVE_DIR = 'bm3d'

# change parameters as needed
enhancer = LR3M(alpha=0.015, beta=1.5e-3, eps=10, lmbda=2.5, sigma=10)
enhancer.convergence_error = 0.4

for f in os.listdir(IMAGE_DIR):
    fbase = os.path.splitext(f)[0]
    ext = os.path.splitext(f)[1]

    if ext != '.png':
        continue

    im_in = np.array(Image.open(f'{IMAGE_DIR}/{f}'))

    # SET MINIMIZER 
    minimizer = BM3DMinimizer(noise_sigma=4)    
    enhancer.set_minimizer(minimizer)

    L,R = enhancer.estimate(im_in)
    S_improve = enhancer.brighten(L,R,2.2)

    # change this to control the maximum brightness of the adjusted image
    SAT = 1
    output = imagify(S_improve,SAT)
    
    Image.fromarray(output).save(f'{SAVE_DIR}/{fbase}_enhanced.png')

