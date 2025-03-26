import numpy as np
from L3RM.utils import first_order_derivative_filter, get_matrix_gradient

class L3RM:
    def __init__(self):
        pass

    def process(self, 
                input_img: np.ndarray
                ):
        self._initialize(input_img)


    def _initialize(self, 
                    input_img: np.ndarray
                    ):
        # initial illumination map is the average of 3 channels
        self.lhat = np.mean(input_img,2)
        
        # construct derivative filters based on image size
        [height,width,channels] = input_img.shape
        self.dx = first_order_derivative_filter(width, -1)
        self.dy = first_order_derivative_filter(height)

        



