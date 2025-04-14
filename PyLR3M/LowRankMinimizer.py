import numpy as np
from dataclasses import dataclass
from typing import Tuple

from PyLR3M.utils import soft_shrinkage
from PyLR3M.MinimizerBase import MinimizerBase

@dataclass
class LowRankMinParams:
    block_size:     int     =   21                  # block size
    match_window:   int     =   5                  # maximum distance of blocks to consider
    step:           int     =   10                   # number of pixels between block centers
    kNN:            int     =   10                  # number of nearest neighbours

    num_iters:      int     =   10                  # number of iterations for denoising
    std_prior:      float   =   10                  # prior on the noise variance
    delta:          float   =   0.23                # noise reintroduction regularization parameter
    gamma:          float   =   0.65                # variance re-estimation scaling
    c1:             float   =   2.9*np.sqrt(2)      # soft shrinkage threshold scaling parameter


class LowRankMinimizer(MinimizerBase):
    def __init__(self):
        self.params = LowRankMinParams()

    def minimize(self, 
                 im_in: np.ndarray
                 ):
        self._init_indexing(im_in)
        im_out = np.copy(im_in)

        for self.iter in range(self.params.num_iters):
            im_out = self._regularization(im_in,im_out)
            self._var_estimation(im_in,im_out)
            X = self._patchify(im_out,self.params.block_size)
            kNN = self._block_matching(X)
            im_out = self._low_rank_SSC(X,kNN)

            print(f"completed iter {self.iter}")

        return im_out
            
    def _init_indexing(self,
                       im_in: np.ndarray
                       ):
        '''
        Initializes the indexing used for block matching and SVD
        '''
        # save the image size parameters
        [self.n,self.m,self.C] = im_in.shape
        self.N = self.n - self.params.block_size + 1
        self.M = self.m - self.params.block_size + 1
        self.L = (self.N)*(self.M)

        # linearly index every patch
        self.patch_index = np.reshape(np.arange(self.L),(self.N,self.M)) 

        # create selection matrix depending on the patch step
        I2 = np.zeros((self.N,self.M))
        I2[::self.params.step,::self.params.step] = 1

        # get the indexes of patches used for block matching
        [self.patch_ind_row,self.patch_ind_col] = np.nonzero(I2)
        self.num_blocks = len(self.patch_ind_row)
        print(self.num_blocks)

    def _regularization(self,
                        im_in,
                        im_out
                        ) -> np.ndarray:
        '''
        Reintroduce noise from the original image for regularization.
        '''
        return im_out + self.params.delta*(im_in-im_out)
        
    def _var_estimation(self,
                        im_in,
                        im_out
                        ):
        '''
        Estimates the noise variance
        '''
        error = im_out - im_in
        noise_error = self.params.std_prior**2-np.mean(np.pow(error,2))
        self.sig_w = np.sqrt(np.abs(noise_error))
        if self.iter != 0:
            self.sig_w *= self.params.gamma
            
    def _patchify(self, 
                  im_in: np.ndarray, 
                  block_size: int
                  )->np.ndarray:
        # flatten every [fxfxC] patch into a row in X
        X = np.zeros((int(block_size**2)*self.C,self.L))
        for i in range(block_size):
            for j in range(block_size):
                for k in range(self.C):
                    add_block = im_in[i:self.n-block_size+i+1,
                                      j:self.m-block_size+j+1,
                                      k].flatten()
                    X[i*block_size+j*self.C+k,:] = add_block
        return X.T

    def _block_matching(self,
                        patches:    np.ndarray
                        )-> np.ndarray:
        '''
        Returns the k nearest neighbours of each patch.
        '''
        f = self.params.block_size
        self.block_elems = f**2*3

        kNN = np.zeros((self.params.kNN,self.num_blocks))
        for i in range(self.num_blocks):
            if (i+1) % 100 == 0:
                print(f"matched block {i+1}/{self.num_blocks}")
            (r,c) = (self.patch_ind_row[i],self.patch_ind_col[i])
            offset = r*self.M+c

            block_match_rmin = int(np.max([r-self.params.match_window,0]))
            block_match_rmax = int(np.min([r+self.params.match_window,self.N]))
            block_match_cmin = int(np.max([c-self.params.match_window,0]))
            block_match_cmax = int(np.min([c+self.params.match_window,self.M]))

            match_idxs = self.patch_index[block_match_rmin:block_match_rmax,
                                          block_match_cmin:block_match_cmax]
            match_idxs = match_idxs.flatten()
            num_matches = len(match_idxs)

            exemplar = patches[offset,:]
            exemplar = np.tile(exemplar,(num_matches,1))
            blocks = patches[match_idxs,:]

            error = exemplar - blocks
            l2_error = np.linalg.norm(error,2,axis=1)/self.block_elems

            sorted_inds = np.argsort(l2_error)
            kNN[:,i] = match_idxs[sorted_inds[:self.params.kNN]]

        return kNN.astype(int)

    def _low_rank_SSC(self,
                      patches: np.ndarray, 
                      kNN: np.ndarray
                      ):
        Y = np.zeros(patches.shape)
        W = np.zeros(patches.shape)

        for i in range(self.num_blocks):
            # print(kNN[:,i])
            blocks = patches[kNN[:,i],:]
            n_blocks = blocks.shape[0]
            block_means = np.tile(np.mean(blocks,0),(n_blocks,1))
            blocks = blocks - block_means

            [Y[kNN[:,i],:],W[kNN[:,i],:]] = self._block_SSC(blocks,
                                                            block_means)
        return self._unpatchify(Y,W)

    def _block_SSC(self,
                   blocks: np.ndarray,
                   block_means: np.ndarray
                   )->Tuple[np.ndarray,np.ndarray]:
        [U,sigma,Vh] = np.linalg.svd(blocks, full_matrices=False)
        
        sig_i = np.maximum(np.pow(sigma,2)/self.block_elems - self.sig_w**2, 0)
        tau = self.params.c1*self.sig_w**2 / (np.sqrt(sig_i) + np.finfo(float).eps)
        sig_i = soft_shrinkage(sig_i,tau)

        r = np.sum(sig_i>0)

        U = U[:,1:r]
        Vh = Vh[1:r,:]
        X = U @ np.diag(sig_i[1:r]) @ Vh

        if r == self.L:
            weight = 1/self.L
        else:
            weight = (self.L-r)/self.L
        W = weight*np.ones(X.shape)
        X = (X + block_means)*weight

        return X,W

    def _unpatchify(self,
                    Y: np.ndarray,
                    W: np.ndarray
                    ):
        im_out = np.zeros((self.n,self.m,self.C))
        im_weights = np.zeros((self.n,self.m,self.C))

        for i in range(self.params.block_size):
            for j in range(self.params.block_size):
                for k in range(self.C):
                    lin_index = (i*self.params.block_size)+(j*self.C)+k
                    im_out[i:self.n-self.params.block_size+i+1,
                           j:self.m-self.params.block_size+j+1,
                           k] = \
                        np.reshape(Y[:,lin_index].T,(self.N,self.M))
                    im_weights[i:self.n-self.params.block_size+i+1,
                               j:self.m-self.params.block_size+j+1,
                               k] = \
                        np.reshape(W[:,lin_index].T,(self.N,self.M))

        return im_out / im_weights