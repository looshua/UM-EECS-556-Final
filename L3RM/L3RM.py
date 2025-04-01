import numpy as np
from L3RM.utils import first_order_derivative_filter, get_matrix_gradient
from L3RM.LowRankMinimizer import LowRankMinimizer

class L3RM:
    def __init__(self, 
                 alpha: float,      # luminance L1 norm penalty
                 beta:  float,      # reflectance - gradient Frobenius norm penalty
                 rho:   float,      # penalty scalar adjustment factor
                 lmbda: float,      # gradient amplification factor for calculating G
                 sigma: float,      # gradient amplification rate for calculating G
                 eps:   float,      # small gradient suppression threshold for calculating G
                 ):
        self.alpha = alpha
        self.beta = beta
        self.rho = rho

        self.lmbda = lmbda
        self.sigma = sigma
        self.eps = eps

        self.minimizer = LowRankMinimizer()
        self.max_R_iters = 5
        self.convergence_error = 5

    def estimate(self, 
                 input_img: np.ndarray
                 ):
        Lhat, Rhat, Gh, Gv = self._initialize(input_img)
        L = self._solve_L_subproblem(Lhat)

        k = 0
        converge = False
        while not converge:
            k += 1
            converge, R = self._solve_R_subproblem(Rhat, Gh, Gv)

        return L, R
    
    def brighten(self, L, R, gamma):
        brighter_L = np.pow(L,gamma)
        return R*brighter_L
            
    def _initialize(self, 
                    input_img: np.ndarray
                    ):
        # initial illumination map is the average of 3 channels
        Lhat = np.mean(input_img,2)
        Rhat = input_img / Lhat
        
        # construct derivative filters based on image size
        [height,width,channels] = input_img.shape
        self.dh = first_order_derivative_filter(width, -1)
        self.dv = first_order_derivative_filter(height)

        self.Dh = first_order_derivative_filter(width*height,-width)
        self.Dv = first_order_derivative_filter(width*height,height)
        self.DtD = (self.Dh.T @ self.Dh) + (self.Dv.T @ self.Dv)

        self.height = height
        self.width = width
        self.channels = channels

        self.mu = 1
        self.Z = np.zeros((height,width))

        Gh, Gv = self._get_G(input_img)

        self.r_iter_k = 0

        return Lhat, Rhat, Gh, Gv
    
    def _get_G(self,input_img):
        delh_S, delv_S = get_matrix_gradient(input_img, self.dh, self.dv)
        Gh = self._calc_G_from_del_S(delh_S)
        Gv = self._calc_G_from_del_S(delv_S)
        return Gh, Gv

    def _calc_G_from_del_S(self,del_S):
        mask = np.abs(del_S) > self.eps
        del_Shat = del_S * mask
        G = (1+self.lmbda*np.exp(np.abs(del_Shat)/self.sigma)) * del_Shat
        return G

    def _solve_L_subproblem(self,
                            Lhat: np.ndarray
                            ):
        delh_Lhat, delv_Lhat = get_matrix_gradient(Lhat, self.dh, self.dv)
        ah = self._get_ad(delh_Lhat)
        av = self._get_ad(delv_Lhat)

        solution_matrix = np.eye(self.height*self.width) + \
                          (self.Dh.T @ ah @ self.Dh) + \
                          (self.Dv.T @ av @ self.Dv)
        lhat = np.reshape(Lhat, -1)
        l = np.linalg.solve(solution_matrix,lhat)

        L = np.reshape(l, (self.height, self.width))
        return L

    def _get_ad(self, deld_Lhat):
        deld_lhat = np.reshape(deld_Lhat,-1)
        ad = np.diag(self.alpha/(np.abs(deld_lhat)+1e-10))
        return ad

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
        solution_matrix = 2*self.beta*self.DtD + \
                          self.mu*np.eye(self.height*self.width)

        rhat = np.reshape(Rhat,-1)
        z = np.reshape(self.Z,-1)

        gh = np.reshape(Gh, -1)
        gv = np.reshape(Gv, -1)
        solution_vector = 2*self.beta*((self.Dh @ gh) + (self.Dv @ gv)) + \
                          self.mu*rhat + z
        
        rhat_kp1 = np.linalg.solve(solution_matrix,solution_vector)
        Rhat_kp1 = np.reshape(rhat_kp1, (self.height,self.width))
        return Rhat_kp1

    def _solve_R_noise_suppression(self, input_img, Rhat_kp1, L):
        Rbar = (2*input_img*L + self.mu*Rhat_kp1 + self.Z) / \
               (2*L*L + self.mu)
        Rkp1 = self.minimizer.minimize(Rbar)
        return Rkp1
        
    def _update_aux(self, Rhat, R):
        self.Z = self.Z + self.mu*(Rhat - R)
        self.mu = self.mu * self.rho

    def check_converge(self, Rhat, R) -> bool:
        if self.r_iter_k > self.max_R_iters:
            return True
        
        error = np.linalg.norm((Rhat - R).flatten(),2)/R.size()
        return error < self.convergence_error




