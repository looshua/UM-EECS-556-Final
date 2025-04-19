import cv2
import numpy as np

from PyLR3M.MinimizerBase import MinimizerBase

class FNLMMinimizer(MinimizerBase):
    def __init__(self, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21):
        self.h = h
        self.hColor = hColor
        self.templateWindowSize = templateWindowSize
        self.searchWindowSize = searchWindowSize

    def minimize(self, im_in: np.ndarray):
        im_in = np.clip(im_in, 0, 3)
        img_min = im_in.min()
        img_max = im_in.max()
        img_scaled = 255 * (im_in - img_min) / (img_max - img_min)
        img_uint8 = img_scaled.astype(np.uint8)
        denoised = cv2.fastNlMeansDenoisingColored(img_uint8, None, self.h, self.hColor, self.templateWindowSize, self.searchWindowSize)
        
        return img_min + (denoised.astype(np.float64) / 255) * (img_max - img_min)
    
