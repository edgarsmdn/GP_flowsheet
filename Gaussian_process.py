"""
GP_flowsheet project

                                Gaussian process

Author   : Antonio del Rio Chanona
Edited by: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
Date: April 2020

Notes on editions (edited lines have the following at the end '#!!!'):
    
    1. The nugget term was added to the covariance matrix to prevent not 
        positive definite matrix.
    2. A security section was added in the negative_loglikelihood function to 
        ensure there are not nan or inf in K. This problem is 
        encountered sometimes during the hyperparameters' optimization. In case
        any nan or inf is present, 10e80 is return as the final value.
    3. A security section was added before the cholesky decomposition to 
        ensure there are not negative eigenvalues in K. This problem is 
        encountered sometimes during the hyperparameters' optimization. In case
        any negative eigenvalue is present, 10e80 is return as the final value.
    4. Bounds were included during the hyperparameter optimization to prevent 
        not positive definite Matrix.
    5. The function 'calc_cov_sample_casadi' was added to replace this calculation
        whenever the function 'GP_inference_casadi' is called.
    6. Introduce the Matern 3/2 and 5/2 kernel.
    
"""
import numpy as np
from scipy.spatial.distance import cdist
import sobol_seq
from scipy.optimize import minimize
from casadi import *

class GP:
    
    ###########################
    # --- initializing GP --- #
    ###########################    
    def __init__(self, X, Y, kernel, multi_hyper):
        
        # GP variable definitions
        self.X, self.Y, self.kernel = X, Y, kernel
        self.n_point, self.nx_dim   = X.shape[0], X.shape[1]
        self.ny_dim                 = Y.shape[1]
        self.multi_hyper            = multi_hyper
        
        # normalize data
        self.X_mean, self.X_std     = np.mean(X, axis=0), np.std(X, axis=0)
        self.Y_mean, self.Y_std     = np.mean(Y, axis=0), np.std(Y, axis=0)
        self.X_norm, self.Y_norm    = (X-self.X_mean)/self.X_std, (Y-self.Y_mean)/self.Y_std
        
        # determine hyperparameters
        self.hypopt, self.invKopt   = self.determine_hyperparameters()        
    
    #############################
    # --- Covariance Matrix --- #
    #############################    
    
    def Cov_mat(self, kernel, X_norm, W, sf2):
        '''
        Calculates the covariance matrix of a dataset Xnorm
        --- decription ---
        '''
        nugget     = np.eye(X_norm.shape[0])* 1.1e-6                        #!!!
        if kernel == 'RBF':
            dist       = cdist(X_norm, X_norm, 'seuclidean', V=W)**2
            cov_matrix = sf2*np.exp(-0.5*dist) + nugget                     #!!!
            # Note: cdist =>  sqrt(sum(u_i-v_i)^2/V[x_i])
        elif kernel =='Matern_32':                                          
            dist = cdist(X_norm, X_norm, 'seuclidean', V=W)
            t_1 = 1 + 3**0.5*dist 
            t_2 = np.exp(-3**0.5*dist)
            cov_matrix = sf2 * t_1 * t_2 + nugget
        elif kernel =='Matern_52':                                          
            dist = cdist(X_norm, X_norm, 'seuclidean', V=W)
            t_1  = 1 + 5**0.5* dist + 5/3*dist**2
            t_2  = np.exp(-5**2*dist)
            cov_matrix = sf2 * t_1 * t_2 + nugget
        else:
            print('ERROR no kernel with name ', kernel)
            
        return cov_matrix

    ################################
    # --- Covariance of sample --- #
    ################################    
        
    def calc_cov_sample(self,xnorm,Xnorm,ell,sf2, kernel):
        '''
        Calculates the covariance of a single sample xnorm against the dataset Xnorm
        --- decription ---
        '''    
        # internal parameters
        nx_dim = self.nx_dim
        
        if kernel == 'RBF':
            dist = cdist(Xnorm, xnorm.reshape(1,nx_dim), 'seuclidean', V=ell)**2
            cov_matrix = sf2 * np.exp(-.5*dist)
        elif kernel =='Matern_32':
            dist = cdist(Xnorm, xnorm.reshape(1,nx_dim), 'seuclidean', V=ell)
            t_1 = 1 + 3**0.5*dist 
            t_2 = np.exp(-3**0.5*dist)
            cov_matrix = sf2 * t_1 * t_2
        elif kernel == 'Matern_52':
            dist = cdist(Xnorm, xnorm.reshape(1,nx_dim), 'seuclidean', V=ell)
            t_1  = 1 + 5**0.5* dist + 5/3*dist**2
            t_2  = np.exp(-5**2*dist)
            cov_matrix = sf2 * t_1 * t_2
        else:
            print('ERROR no kernel with name ', kernel)

        return cov_matrix 

    def calc_cov_sample_casadi(self, xnorm, Xnorm, ell, sf2, kernel): #!!!
        '''
        Calculates covariance of single sample against data set using CASADI notation
        '''
        if kernel == 'RBF':
            dist = sum2((Xnorm - repmat(xnorm, len(Xnorm)))**2 /
                    repmat(transpose(ell), len(Xnorm)))
            cov_matrix = sf2 * exp(-.5*dist)
        elif kernel =='Matern_32':
            dist = sqrt(sum2((Xnorm - repmat(xnorm, len(Xnorm)))**2 /
                    repmat(transpose(ell), len(Xnorm))))
            t_1 = 1 + 3**0.5*dist 
            t_2 = exp(-3**0.5*dist)
            cov_matrix = sf2 * t_1 * t_2
        elif kernel == 'Matern_52':
            dist = sqrt(sum2((Xnorm - repmat(xnorm, len(Xnorm)))**2 /
                    repmat(transpose(ell), len(Xnorm))))
            t_1  = 1 + 5**0.5* dist + 5/3*dist**2
            t_2  = exp(-5**2*dist)
            cov_matrix = sf2 * t_1 * t_2
        else:
            print('ERROR no kernel with name ', kernel)
        
        return cov_matrix           
        
    ###################################
    # --- negative log likelihood --- #
    ###################################   
    
    def negative_loglikelihood(self, hyper, X, Y):
        '''
        --- decription ---
        ''' 
        # internal parameters
        n_point, nx_dim = self.n_point, self.nx_dim
        kernel          = self.kernel
        
        W               = np.exp(2*hyper[:nx_dim])   # W <=> 1/lambda
        sf2             = np.exp(2*hyper[nx_dim])    # variance of the signal 
        sn2             = np.exp(2*hyper[nx_dim+1])  # variance of noise

        K       = self.Cov_mat(kernel, X, W, sf2)  # (nxn) covariance matrix (noise free)
        K       = K + (sn2 + 1e-8)*np.eye(n_point) # (nxn) covariance matrix
        K       = (K + K.T)*0.5                    # ensure K is simetric
        
        #print(np.linalg.eig(K)[0])                 # Print eigenvalues of K
        # Ensure there is no inf or nan in K                                #!!!
        if np.any(np.isnan(K)) or np.any(np.isinf(K)):
            return 10e80
        # Ensure K has no negative eigenvalues                              #!!!
        if np.any(np.linalg.eig(K)[0] < 0):
            return 10e80
        L       = np.linalg.cholesky(K)            # do a cholesky decomposition
        logdetK = 2 * np.sum(np.log(np.diag(L)))   # calculate the log of the determinant of K the 2* is due to the fact that L^2 = K
        invLY   = np.linalg.solve(L,Y)             # obtain L^{-1}*Y
        alpha   = np.linalg.solve(L.T,invLY)       # obtain (L.T L)^{-1}*Y = K^{-1}*Y
        NLL     = np.dot(Y.T,alpha) + logdetK      # construct the NLL

        return NLL
    
    ############################################################
    # --- Minimizing the NLL (hyperparameter optimization) --- #
    ############################################################   
    
    def determine_hyperparameters(self):
        '''
        --- decription ---
        Notice we construct one GP for each output
        '''   
        # internal parameters
        X_norm, Y_norm  = self.X_norm, self.Y_norm
        nx_dim, n_point = self.nx_dim, self.n_point
        kernel, ny_dim  = self.kernel, self.ny_dim
        Cov_mat         = self.Cov_mat
        
        
        lb               = np.array([-3.]*(nx_dim+1) + [-8.])  # lb on parameters (this is inside the exponential)
        ub               = np.array([3.]*(nx_dim+1) + [ 4.])   # lb on parameters (this is inside the exponential)
        bounds           = np.hstack((lb.reshape(nx_dim+2,1),
                                      ub.reshape(nx_dim+2,1)))
        
        #bounds           = [[1, 3], [1, 3], [1, 3]]
        
        multi_start      = self.multi_hyper                   # multistart on hyperparameter optimization
        multi_startvec   = sobol_seq.i4_sobol_generate(nx_dim + 2,multi_start)

        options  = {'disp':False,'maxiter':10000}             # solver options
        hypopt   = np.zeros((nx_dim+2, ny_dim))               # hyperparams w's + sf2+ sn2 (one for each GP i.e. output var)
        localsol = [0.]*multi_start                           # values for multistart
        localval = np.zeros((multi_start))                    # variables for multistart

        invKopt = []
        # --- loop over outputs (GPs) --- #
        for i in range(ny_dim):    
            # --- multistart loop --- # 
            for j in range(multi_start):
                #print('multi_start hyper parameter optimization iteration = ',j,'  input = ',i)
                hyp_init    = lb + (ub-lb)*multi_startvec[j,:]
                # --- hyper-parameter optimization --- #
                res = minimize(self.negative_loglikelihood,hyp_init,args=(X_norm,Y_norm[:,i])\
                               ,method='SLSQP',options=options,tol=1e-12, bounds=bounds) #!!!
                #print(' Initial guess for hyperparameters: ',hyp_init )
                #print(' Bounds for hyper parameters: ', bounds)
                #print(' Optimal hyperparameters: ', res.x)
                #print(' Optimal Objective function value: ', res.fun )
                localsol[j] = res.x
                localval[j] = res.fun

            # --- choosing best solution --- #
            minindex    = np.argmin(localval)
            hypopt[:,i] = localsol[minindex]
            ellopt      = np.exp(2.*hypopt[:nx_dim,i])
            sf2opt      = np.exp(2.*hypopt[nx_dim,i])
            sn2opt      = np.exp(2.*hypopt[nx_dim+1,i]) + 1e-8

            # --- constructing optimal K --- #
            Kopt        = Cov_mat(kernel, X_norm, ellopt, sf2opt) + sn2opt*np.eye(n_point)
            # --- inverting K --- #
            invKopt     += [np.linalg.solve(Kopt,np.eye(n_point))]

        return hypopt, invKopt

    ########################
    # --- GP inference --- #
    ########################     
    
    def GP_inference_np(self, x):
        '''
        --- decription ---
        '''
        nx_dim                   = self.nx_dim
        kernel, ny_dim           = self.kernel, self.ny_dim
        hypopt, Cov_mat          = self.hypopt, self.Cov_mat
        stdX, stdY, meanX, meanY = self.X_std, self.Y_std, self.X_mean, self.Y_mean
        calc_cov_sample          = self.calc_cov_sample
        invKsample               = self.invKopt
        Xsample, Ysample         = self.X_norm, self.Y_norm
        # Sigma_w                = self.Sigma_w (if input noise)

        xnorm = (x - meanX)/stdX
        mean  = np.zeros(ny_dim)
        var   = np.zeros(ny_dim)
        # --- Loop over each output (GP) --- #
        for i in range(ny_dim):
            invK           = invKsample[i]
            hyper          = hypopt[:,i]
            ellopt, sf2opt = np.exp(2*hyper[:nx_dim]), np.exp(2*hyper[nx_dim])
            # --- determine covariance of each output --- #
            k       = calc_cov_sample(xnorm,Xsample,ellopt,sf2opt,kernel)
            mean[i] = np.matmul(np.matmul(k.T,invK),Ysample[:,i])
            var[i]  = sf2opt - np.matmul(np.matmul(k.T,invK),k)
            #var[i] = sf2opt + Sigma_w[i,i]/stdY[i]**2 - np.matmul(np.matmul(k.T,invK),k) (if input noise)

        # --- compute un-normalized mean --- #    
        mean_sample = mean*stdY + meanY
        var_sample  = var*stdY**2

        return mean_sample, var_sample
    
    def GP_inference_casadi(self, x): #!!!
        '''
        --- decription ---
        '''
        nx_dim                   = self.nx_dim
        kernel, ny_dim           = self.kernel, self.ny_dim
        hypopt, Cov_mat          = self.hypopt, self.Cov_mat
        stdX, stdY, meanX, meanY = self.X_std.reshape(1,-1), self.Y_std, self.X_mean.reshape(1,-1), self.Y_mean
        calc_cov_sample          = self.calc_cov_sample_casadi
        invKsample               = self.invKopt
        Xsample, Ysample         = self.X_norm, self.Y_norm
        # Sigma_w                = self.Sigma_w (if input noise)

        xnorm = (x - meanX)/stdX
        # --- Loop over each output (GP) --- #
        for i in range(ny_dim):
            if ny_dim != 1:
                raise Exception ('Current implementation only supports a single-output GP')
            invK           = invKsample[i]
            hyper          = hypopt[:,i]
            ellopt, sf2opt = np.exp(2*hyper[:nx_dim]), np.exp(2*hyper[nx_dim])
            # --- determine covariance of each output --- #
            k       = calc_cov_sample(xnorm, Xsample, ellopt, sf2opt,kernel)
            mean    = (k.T @ invK) @ Ysample[:,i].reshape(-1,1)
            var     = sf2opt - (k.T @ invK) @ k
            #var[i] = sf2opt + Sigma_w[i,i]/stdY[i]**2 - np.matmul(np.matmul(k.T,invK),k) (if input noise)

        # --- compute un-normalized mean --- #    
        mean_sample = mean*stdY + meanY
        var_sample  = var*stdY**2

        return mean_sample, var_sample