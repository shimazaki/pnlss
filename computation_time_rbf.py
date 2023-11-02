#CUDA_VISIBLE_DEVICES=1 python computation_time_rbf.py 

import numpy
import matplotlib.pyplot as plt

import sys
import os

# Calculating computatin time
import time

from jax.config import config
config.update("jax_enable_x64", True)
from jax import numpy as jnp

from gaussian_toolbox import timeseries
from gaussian_toolbox.timeseries import state_model, observation_model, ssm


# Settings
# Noise
numpy.random.seed(0)
noise_level = .01 #1.

#EM convergence criteria
CONV_CRIT = -1e-10 #1e-4
MAX_ITER = 50


# CONV_CRIT = 1e-5 #1e-4
# MAX_ITER = 100

from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

from scipy.integrate import solve_ivp
from scipy.stats import zscore


REPEAT = 10
Dk_range = range(5,41,5)
computation_time = numpy.zeros((REPEAT,len(Dk_range)))
dimension = numpy.zeros((REPEAT,len(Dk_range)))
likelihood = numpy.zeros((REPEAT,len(Dk_range)))
param_list = numpy.zeros((REPEAT,len(Dk_range)))


for repeat in range(REPEAT):
    print(repeat)

    # Data generation

    rho = 28.0
    sigma = 10.0
    beta = 8.0 / 3.0

    def lorenz(state, t):
        x, y, z = state  # Unpack the state vector
        return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives

    state0 = [1.0, 1.0, 1.0]
    t = numpy.linspace(0, 30, 300) #short trial

    states = odeint(lorenz, state0, t)

    X = states

    X = zscore(X, axis=0)
    X0 = X.copy()
    X += noise_level * numpy.random.randn(*X.shape)


    # Model fitting
    import numpy
    numpy.random.seed(0)
    import matplotlib.pyplot as plt
    import sys
    import os

    from jax.config import config
    config.update("jax_enable_x64", True)
    from jax import numpy as jnp

    from gaussian_toolbox import timeseries
    from gaussian_toolbox.timeseries import state_model, observation_model, ssm


    Dx = 3
    Dz = 3


    k = 0
    for Dk in Dk_range:
        print(Dk)
        param_list[repeat][k]=Dk
        
        # NLSS
        om = observation_model.LinearObservationModel(Dx, Dz)
        #om.pca_init(X)
        #sm = state_model.LSEMStateModel(Dz, Dk)
        sm = state_model.LRBFMStateModel(Dz, Dk)
        
        ssm_emd = ssm.StateSpaceModel(jnp.array(X), observation_model=om, state_model=sm, max_iter=MAX_ITER, conv_crit=CONV_CRIT)
        
        tc = time.time()
        ssm_emd.fit()
        
        computation_time[repeat][k] = (time.time() - tc)
        print('lorentz in %f s' %(time.time() - tc))
        
        likelihood[repeat][k] = ssm_emd.llk_list[-1]
        
        #state 
        A_dim = numpy.prod(ssm_emd.sm.A.shape) 
        b_dim = numpy.prod(ssm_emd.sm.b.shape)
        mu_dim = numpy.prod(ssm_emd.sm.mu.shape)
        log_length_scale_dim = numpy.prod(ssm_emd.sm.log_length_scale.shape)

        tmp = ssm_emd.sm.Qz.shape[0]
        tmp = int(tmp + tmp*(tmp-1)/2)
        Qz_dim = tmp
        
        DIM = A_dim + b_dim + mu_dim + log_length_scale_dim + Qz_dim
        
        # observation
        C_dim = numpy.prod(ssm_emd.om.C.shape) 
        d_dim = numpy.prod(ssm_emd.om.d.shape)
        tmp = ssm_emd.om.Qx.shape[0]
        tmp = int(tmp + tmp*(tmp-1)/2)
        Qx_dim = tmp
        
        DIM += C_dim + d_dim + Qx_dim
        dimension[repeat][k]=DIM

        k += 1


# Saving the data
import pickle
f = open('lorentz_computation_time_rbf_fix.dat','wb')
#f = open('lorentz_computation_time_rbf_cri.dat','wb')
pickle.dump((param_list, computation_time, dimension, likelihood), f)
f.close
