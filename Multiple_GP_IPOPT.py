"""
GP_flowsheet project

                        Multiple functions optimization 
                          using GPs and trust-regions
                                 using IPOPT 

Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de
Date: June 2020
"""

import numpy as np
from plot_utilities import plot, meshgrid
import sobol_seq
from Gaussian_process import GP
from scipy.optimize import minimize
from casadi import Opti, norm_2

###########################
# --- Matyas function --- #
###########################

def Matyas(x, y):
    '''
    Matyas function for 2D
    
    Parameters
    ----------
    x : np.array
        Input variables.

    Returns
    -------
    f : np.array
        Matyas output values.
    '''
    f = 0.26*(x**2 + y**2) - 0.48*x*y
    return f/30

# Search space bounds
lb_M = np.array([-10, -1])
ub_M = np.array([2, 10])

###############################
# --- Rosenbrock function --- #
###############################

def Rosenbrock(x, y):
    '''
    Rosenbrock function for 2D
    
    Parameters
    ----------
    x : np.array
        Input variables.

    Returns
    -------
    f : np.array
        Rosenbrock output values.
    '''
    f = (1-x)**2 + 100*(y-x**2)**2
    return f

# Search space bounds
lb_R = np.array([-1.5, -0.5])
ub_R = np.array([1.5, 2.5])

####################
# --- Meshgrid --- #
####################

X_M, Y_M, Z_M = meshgrid(Matyas, lb_M, ub_M)
X_R, Y_R, Z_R = meshgrid(Rosenbrock, lb_R, ub_R)
xM            = np.linspace(lb_M[0], ub_M[0], 100)
xR            = np.linspace(lb_R[0], ub_R[0], 100)

############################################
# --- Determination of feasible region --- #
############################################
bounds   = [(lb_M[0], ub_M[0]), (lb_M[1], ub_M[1])] 
x0       = (-2,8)

def Feasible_region(x):
    f = Matyas(x[0], x[1])
    return f

def Const_1M(x):
    return x[1] - 5*x[0]

def Const_2M(x):
    return x[1] - 0.03*x[0]**2


res        = minimize(Feasible_region, x0, method='SLSQP', bounds=bounds, 
                      constraints=({"fun": Const_1M, "type": "ineq"},
                                   {"fun": Const_2M, "type": "ineq"}))
Min_Matyas = res.fun

###################################
# --- Trust-regions algorithm --- #
###################################

# 1. Initial trust-regions (it has to be a feasible point for the complete system)
# ----------------------------------------------------------------------------

x_centerM = np.array([-8, 8]).reshape(1,-1)
x_centerR = np.array([-1, Matyas(x_centerM[0,0], x_centerM[0,1])]).reshape(1,-1)

radius_M = 0.5
radius_R = 0.3

# 2. Sample within the TRs
# ----------------------------------------------------------------------------

def rescale_TR_Sobol(center, radius, sobol):
    '''
    This re-scaling is taking place from the Sobol sequence to the square with 
    lenghtsize equal to the radius and centered in the specified center. A 
    possible improvement here will be to use DoE in polar coordinates instead 
    (e.g https://hal.archives-ouvertes.fr/hal-01119942v2/document).
    '''
    ub_TR = center + radius                 # Upperbounds of square
    lb_TR = center - radius                 # Lowerbounds of square
    X_TR  = lb_TR + (ub_TR - lb_TR)*sobol
    return X_TR

# -- Matyas
n_samples_M = 20
n_dim_M     = x_centerM.shape[1]
Sobol_TR_M  = sobol_seq.i4_sobol_generate(n_dim_M, n_samples_M)

Xtrain_M_org = rescale_TR_Sobol(x_centerM, radius_M, Sobol_TR_M)
Ytrain_M_org = Matyas(Xtrain_M_org[:,0], Xtrain_M_org[:,1]).reshape(-1, 1)

# -- Rosenbrock
n_samples_R = 20
n_dim_R     = x_centerR.shape[1]
Sobol_TR_R  = sobol_seq.i4_sobol_generate(n_dim_R, n_samples_R)

Xtrain_R_org = rescale_TR_Sobol(x_centerR, radius_R, Sobol_TR_R)
Ytrain_R_org = Matyas(Xtrain_R_org[:,0], Xtrain_R_org[:,1]).reshape(-1, 1)

# 3. Optimization loop
# ----------------------------------------------------------------------------

def Obj_fun_TR(d, GP_instance, center):
    dR    = d[2:]
    mean  = GP_instance.GP_inference_casadi(center + dR)[0]
    std   = GP_instance.GP_inference_casadi(center + dR)[1]
    alpha = 3
    f     =  mean #+ alpha*std
    return f

# -- Optimization specifications
n_opt_TR = 50      

eta_1    = 0.1          # eta 1
eta_2    = 0.1          # eta 2
eta_3    = 0.9          # eta 3
k_0      = 0.5          # shrinkage constant
k_1      = 2            # expansion constant
radmax   = 0.9          # maximum TR radius
lb       = np.array([-radius_M, -radius_M, -radius_R, -radius_R]).reshape(1,4)
ub       = np.array([radius_M, radius_M, radius_R, radius_R]).reshape(1,4)
eps_TR   = 1e-6         # radius/trust-region tolerance

# -- Get a copy of the original sampling data of TRs
Xtrain_M = np.copy(Xtrain_M_org); Ytrain_M = np.copy(Ytrain_M_org)
Xtrain_R = np.copy(Xtrain_R_org); Ytrain_R = np.copy(Ytrain_R_org)

# -- Plots

plM = plot()
plM.contour(X_M, Y_M, Z_M, title='Matyas function')
plM.scatter(0, 0, label='Global minimum', color='r', marker='o')
plM.line(xM, 5*xM)                                              
plM.line(xM, 0.03*xM**2)
plM.xylim((lb_M[0], ub_M[0]), (lb_M[1], ub_M[1])) 


plR = plot()
plR.contour(X_R, Y_R, np.log(Z_R), title='Rosenbrock function')
plR.scatter(1, 1, label='Global minimum', color='r', marker='o')
plR.line(xR, 2-xR)                                              
plR.line(xR, 2*xR - 1)
plR.line(xR, np.repeat(Min_Matyas, len(xR)), color='r', ls='-.')
plR.xylim((lb_R[0], ub_R[0]), (lb_R[1], ub_R[1]))
                                         

rhol = []
Acl  = []
Prl  = []
trajectoryM = np.zeros((n_opt_TR, 2))
trajectoryR = np.zeros((n_opt_TR, 2))
dM          = np.zeros((n_opt_TR, 2))
dR          = np.zeros((n_opt_TR, 2))
radiiM      = np.zeros(n_opt_TR)
radiiR      = np.zeros(n_opt_TR)

for i in range(n_opt_TR):
    
    # -- 4.1 Train GPs
    GP_modelM = GP(Xtrain_M, Ytrain_M, 'RBF', multi_hyper=3)
    GP_modelR = GP(Xtrain_R, Ytrain_R, 'RBF', multi_hyper=3)
    
    # -- 4.2 Constraint optimization
    opti  = Opti()                                       # Initialize Opti utility
    d     = opti.variable(1,4)                           # Initialize decision variable
    d0    = np.array([radius_M/10, radius_M/10, radius_R/10, radius_R/10])       # Set initial guess
    
    opti.set_initial(d, d0 )                             # Asign initial guess
    opti.minimize(Obj_fun_TR(d, GP_modelR, x_centerR))   # Asign objective function
    
    opti.subject_to( norm_2(d[:2])          <= radius_M)                                        # Constraint TR_M
    opti.subject_to( norm_2(d[2:])          <= radius_R)                                        # Constraint TR_R
    opti.subject_to( x_centerM[0,1] + d[1]  >=  5*(x_centerM[0,0] + d[0]))                                # Constraint 1M
    opti.subject_to( x_centerM[0,1] + d[1]  >=  0.03*(x_centerM[0,0] + d[0])**2)                                # Constraint 2M
    opti.subject_to( x_centerR[0,1] + d[3]  <=  2 - x_centerR[0,0] + d[2])                                # Constraint 1R
    opti.subject_to( x_centerR[0,1] + d[3]  >=  2*(x_centerR[0,0] + d[2]) - 1)                                # Constraint 2R
    opti.subject_to( x_centerR[0,1] + d[3]  == GP_modelM.GP_inference_casadi(x_centerM + d[:2])[0] ) # Constraint connection
    
    opti.subject_to(opti.bounded(lb, d, ub)) # Bounds TR-square
    
    opti.subject_to( x_centerM[0,0] + d[0] >= lb_M[0])
    opti.subject_to( x_centerM[0,0] + d[0] <= ub_M[0])
    opti.subject_to( x_centerM[0,1] + d[1] >= lb_M[1])
    opti.subject_to( x_centerM[0,1] + d[1] <= ub_M[1])
    
    opti.subject_to( x_centerR[0,0] + d[2] >= lb_R[0])
    opti.subject_to( x_centerR[0,0] + d[2] <= ub_R[0])
    opti.subject_to( x_centerR[0,1] + d[3] >= lb_R[1])
    opti.subject_to( x_centerR[0,1] + d[3] <= ub_R[1])
    
    dict_options = {'ipopt.print_level':3, 'ipopt.max_iter':100000, 
                    'ipopt.linear_solver':'mumps'}
    opti.solver('ipopt', dict_options )
    
    try:
        sol = opti.solve()
        d_new = sol.value(d)
        
        # ---- Plots
        plM.TR_plot(x_centerM.reshape(-1,), radius_M)
        plM.scatter(x_centerM[0,0] + d_new[0], x_centerM[0,1] + d_new[1], markersize=15, 
                      marker='o', color='b', label='Current optimum')
        
        plR.TR_plot(x_centerR.reshape(-1,), radius_R)
        plR.scatter(x_centerR[0,0] + d_new[2], x_centerR[0,1] + d_new[3], markersize=15, 
                      marker='o', color='b', label='Current optimum')
    
    except:
        d_new = opti.debug.value(d)
        # ---- Plots
        plM.TR_plot(x_centerM.reshape(-1,), radius_M, color='r')
        plM.scatter(x_centerM[0,0] + d_new[0], x_centerM[0,1] + d_new[1], markersize=15, 
                      marker='o', color='r', label='Current optimum')
        
        plR.TR_plot(x_centerR.reshape(-1,), radius_R, color='r')
        plR.scatter(x_centerR[0,0] + d_new[2], x_centerR[0,1] + d_new[3], markersize=15, 
                      marker='o', color='r', label='Current optimum')
        
    # ---- Save trajectory
    trajectoryM[i] = (x_centerM[0,0], x_centerM[0,1])
    trajectoryR[i] = (x_centerR[0,0], x_centerR[0,1])
    dM[i]          = (d_new[0], d_new[1])
    dR[i]          = (d_new[2], d_new[3])
    radiiM[i]      = radius_M
    radiiR[i]      = radius_R
    
    # -- 4.3 Calculate rho (ratio of the actual cost reduction to the predicted cost reduction)
    Actual_cr    = Rosenbrock(x_centerR[0,0], x_centerR[0,1]) - Rosenbrock(x_centerR[0,0] + d_new[2], x_centerR[0,1] + d_new[3])
    Predicted_cr = GP_modelR.GP_inference_np(x_centerR)[0] - GP_modelR.GP_inference_np(x_centerR + d_new[2:])[0]
    
    rho          = Actual_cr / Predicted_cr
    
    Acl.append(Actual_cr)
    Prl.append(Predicted_cr)
    rhol.append(rho)
    
    # -- 4.4 Update TRs radius according to rho
    if rho < eta_2:
        radius_M   = radius_M                       # Keep TR radius
        radius_R   = radius_R*k_0                   # Shrink TR radius
        
    elif rho > eta_3 and (radius_R - np.linalg.norm(d_new[2:],2) ) < eps_TR and (radius_M - np.linalg.norm(d_new[:2],2) ) < eps_TR:
        radius_M   = min(radius_M*k_1, radmax)      # Exapnd TR radius
        radius_R   = min(radius_R*k_1, radmax)      # Exapnd TR radius
    
    else:
        radius_M   = radius_M                       # Keep TR radius
        radius_R   = radius_R                       # Keep TR radius
    
    # -- 4.5 Update TRs center according to rho
    if rho < eta_1:
        x_centerM = x_centerM                       # Refuse d
        x_centerR = x_centerR                       # Refuse d
    else:
        x_centerM = x_centerM + d_new[:2]           # Accept d
        x_centerR = x_centerR + d_new[2:]           # Accept d
    
    # -- 4.6 Sample within new TR
    Xtrain_M = rescale_TR_Sobol(x_centerM, radius_M, Sobol_TR_M)
    Ytrain_M = Matyas(Xtrain_M[:,0], Xtrain_M[:,1]).reshape(-1, 1)
    
    Xtrain_R = rescale_TR_Sobol(x_centerR, radius_R, Sobol_TR_R)
    Ytrain_R = Rosenbrock(Xtrain_R[:,0], Xtrain_R[:,1]).reshape(-1, 1)
    
    # Print optimization status
    print('\r Optimizing TR number: ', i+1 ,end='\r')
    #print('############################################################')
    
    
# -- Show plots
plM.show(legend_b=False)
plR.show(legend_b=False)

# -- Optimization trajectory GIF
G_M = plot()
G_M.xylim((lb_M[0], ub_M[0]), (lb_M[1], ub_M[1]))
constsM = [(xM, 5*xM), (xM, 0.03*xM**2)]
G_M.gif(trajectoryM, dM, radiiM, X_M, Y_M, Z_M, constsM, title='Matyas function', name='IPOPT_M.gif')


G_R = plot()
G_R.xylim((lb_R[0], ub_R[0]), (lb_R[1], ub_R[1])) 
constsR = [(xR, 2-xR), (xR, 2*xR - 1), (xR, np.repeat(Min_Matyas, len(xR)), 'r', '-.') ]
G_R.gif(trajectoryR, dR, radiiR, X_R, Y_R, np.log(Z_R), constsR, title='Rosenbrock function', name='IPOPT_R.gif')
