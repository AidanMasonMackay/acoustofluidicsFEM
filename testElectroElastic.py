
# This is still under development!

# This code solves for the electric and elastic field
# inside the piezo-electric plate and 'glass'

from electroelastic2d import electroelastic2d
from materialLibrary import piezo

import numpy as np
import os
from numpy import meshgrid
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from collections import OrderedDict

# This code applies a Voltage accross the piezo plate using a Dirichlet condition
# No coupling with the fluid domain yet
# All other boundaries use a zero Neumann condition by default

f = 4.8E6 # actuation frequency
VPiezo = 1 # Voltage applied accross the piezo plate

epsilon = 1e-6 # tolerance when searching for nodes near a particular coordinate

plotPath = f"electroElastic{f*10**(-6)}MHz"
os.makedirs(plotPath, exist_ok=True)

### Build Dirichlet constraints
d_constraints = OrderedDict() # very important this is ordered. This ensures the constraints added last override those added earlier

## Top of piezo plate is at y == 0.0007
def mask_func(x, y):
    return np.isclose(y, 0.0007, atol = epsilon)
def value_func(x, y):
    return VPiezo
d_constraints['d0_pot'] = {'dof_family': 'pot', 
                          'mask_func' : mask_func, 
                          'vals_func': value_func}

# Bottom of piezo plate is at y == 0.0007
def mask_func(x, y):
    return np.isclose(y, 0.0005, atol = epsilon)
def value_func(x, y):
    return  0 # grounding plate

d_constraints['d1_pot'] = {'dof_family': 'pot', 
                          'mask_func' : mask_func, 
                          'vals_func': value_func}


# Set material parameters in each domain
# This is a test and it's a bit messy
# It uses the material parameters of the piezo plate inside the glass

params = {}

params[1] = {
    'eps' : piezo.eps,
    'C' : piezo.C,
    'coupling' : piezo.coupling,
    'rho' : piezo.rho
}
params[2] = {
    'eps' : np.ones_like(piezo.eps),
    'C' : piezo.C,
    'coupling' : np.zeros_like(piezo.coupling),
    'rho' : piezo.rho
}
params[3] = {
    'eps' : np.ones_like(piezo.eps),
    'C' : np.ones_like(piezo.C),
    'coupling' : np.zeros_like(piezo.coupling),
    'rho' : piezo.rho
}

## Working, but with a numerical instability
el = electroelastic2d(domain = [1, 2], params = params, f = f, Dirichlet = d_constraints, meshName = "2DElectroElasticAcoustic/Upto10MHz", builder = 'python')
el.plot(plotPath)
