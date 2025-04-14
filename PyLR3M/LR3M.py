import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from PyLR3M.utils import first_order_derivative_filter, get_matrix_gradient, sparse_place_mat_on_diag
from PyLR3M.LowRankMinimizer import LowRankMinimizer
from PyLR3M.MinimizerBase import MinimizerBase

class LR3M:
    def __init__(self, 
                 alpha: float   =   0.015,          # luminance L1 norm penalty
                 beta:  float   =   0.015,          # reflectance - gradient Frobenius norm penalty
                 rho:   float   =   1.5,            # penalty scalar adjustment factor
                 lmbda: float   =   2.5,            # gradient amplification factor for calculating G
                 sigma: float   =   10,             # gradient amplification rate for calculating G
                 eps:   float   =   1,              # small gradient suppression threshold for calculating G
                 ):
        self.alpha = alpha
        self.beta = beta
        self.rho = rho

        self.lmbda = lmbda
        self.sigma = sigma
        self.eps = eps

        self.minimizer: MinimizerBase = LowRankMinimizer()
        self.max_R_iters = 5
        self.convergence_error = 5

    def set_minimizer(self, minimzer: MinimizerBase):
        self.minimizer = minimzer

    def estimate(self, 
                 input_img: np.ndarray
                 ):
        Lhat, Gh, Gv = self._initialize(input_img)
        L = self._solve_L_subproblem(Lhat)
        Rhat = self._initialize_Rhat(input_img,L)

        k = 0
        converge = False
        while not converge:
            print(f'estimate! {k}')
            k += 1
            converge, R = self._solve_R_subproblem(Rhat, Gh, Gv)

        return L, R
    
    def brighten(self, L, R, gamma):
        brighter_L = np.pow(L,1/gamma)
        brighter_L3 = np.tile(brighter_L[...,np.newaxis],(1,1,3))
        return R*brighter_L3
            
    def _initialize(self, 
                    input_img: np.ndarray
                    ):
        [height,width,channels] = input_img.shape
        self.height = height
        self.width = width
        self.channels = channels

        # initial illumination map is the average of 3 channels
        Lhat = np.mean(input_img,2)
        
        # construct derivative filters based on image size
        self.dh = first_order_derivative_filter(width, -1)
        self.dv = first_order_derivative_filter(height)

        self.Dh = first_order_derivative_filter(width*height,1,is_sparse=True)
        self.Dv = first_order_derivative_filter(width*height,width,is_sparse=True)

        self.DtD = (self.Dh.T @ self.Dh) + (self.Dv.T @ self.Dv)

        self.mu = 1
        self.Z = np.zeros((height,width))

        Gh, Gv = self._get_G(input_img)

        self.r_iter_k = 0

        return Lhat, Gh, Gv
    
    def _get_G(self,
               input_img: np.ndarray):
        Gh = np.zeros(input_img.shape)
        Gv = np.zeros(input_img.shape)

        for c in range(self.channels):
            delh_S, delv_S = get_matrix_gradient(input_img[:,:,c], self.dh, self.dv)
            Gh[:,:,c] = self._calc_G_from_del_S(delh_S)
            Gv[:,:,c] = self._calc_G_from_del_S(delv_S)
        return Gh, Gv

    def _calc_G_from_del_S(self,del_S: np.ndarray):
        mask = np.abs(del_S) > self.eps
        del_Shat = del_S * mask
        G = (1+self.lmbda*np.exp(-np.abs(del_Shat)/self.sigma)) * del_Shat
        return G

    def _solve_L_subproblem(self,
                            Lhat: np.ndarray
                            ):
        delh_Lhat, delv_Lhat = get_matrix_gradient(Lhat, self.dh, self.dv)
        ah = self._get_ad(delh_Lhat)
        av = self._get_ad(delv_Lhat)

        solution_matrix = sparse.eye(self.height*self.width) + \
                          (self.Dh.T @ ah @ self.Dh) + \
                          (self.Dv.T @ av @ self.Dv)
        lhat = np.reshape(Lhat, -1)
        l, info = linalg.cg(solution_matrix,lhat)

        print(info)

        L = np.reshape(l, (self.height, self.width))
        return L

    def _get_ad(self, deld_Lhat):
        deld_lhat = np.reshape(deld_Lhat,-1)
        ad = sparse.diags_array(self.alpha/(np.abs(deld_lhat)+1e-10))
        return ad

    def _initialize_Rhat(self,
                         input_img: np.ndarray,
                         L: np.ndarray
                         ):
        L = L[...,np.newaxis]
        L3 = np.tile(L,(1,1,3)) + 1e-10
        return input_img / L3

    def _solve_R_subproblem(self,
                            input_img,
                            Rhat,
                            Gh,
                            Gv,
                            L
                            ):
        Rhat_kp1 = self._solve_R_contrast_enhancement(Rhat, Gh, Gv)
        R = self._solve_R_noise_suppression(input_img,Rhat_kp1,L)

        self._update_aux()
        converge = self.check_converge(Rhat, R)
        return converge, R

    def _solve_R_contrast_enhancement(self, 
                                      Rhat, 
                                      Gh, 
                                      Gv
                                      ):
        Rhat_kp1 = np.zeros(Rhat.shape)
        for c in range(self.channels):
            solution_matrix = 2*self.beta*self.DtD + \
                            self.mu*sparse.eye(self.height*self.width)

            rhat = np.reshape(Rhat[:,:,c],-1)
            z = np.reshape(self.Z,-1)

            gh = np.reshape(Gh[:,:,c], -1)
            gv = np.reshape(Gv[:,:,c], -1)
            solution_vector = 2*self.beta*((self.Dh.T @ gh) + (self.Dv.T @ gv)) + \
                              self.mu*rhat - z
            
            # rhat_kp1 = np.linalg.solve(solution_matrix,solution_vector)
            rhat_kp1, info = linalg.cg(solution_matrix,solution_vector)
            print(f'R ce converge: {info}')
            Rhat_kp1[:,:,c] = np.reshape(rhat_kp1, (self.height,self.width))
        return Rhat_kp1

    def _solve_R_noise_suppression(self, input_img, Rhat_kp1, L):
        Rbar = self._calc_Rbar(input_img,Rhat_kp1,L)
        Rkp1 = self.minimizer.minimize(Rbar)
        return Rkp1
    
    def _calc_Rbar(self, input_img, Rhat_kp1, L):
        L = L[...,np.newaxis]
        Z = self.Z[...,np.newaxis]
        L3 = np.tile(L,(1,1,3))
        Z3 = np.tile(Z,(1,1,3))

        Rbar = (2*input_img*L3 + self.mu*Rhat_kp1 + Z3) / \
               (2*L3*L3 + self.mu)
        return Rbar
        
    def _update_aux(self, Rhat, R):
        self.Z = self.Z + self.mu*(Rhat - R)
        self.mu = self.mu * self.rho

    def check_converge(self, Rhat, R) -> bool:
        if self.r_iter_k > self.max_R_iters:
            return True
        
        error = np.linalg.norm((Rhat - R).flatten(),2)/R.size()
        return error < self.convergence_error




