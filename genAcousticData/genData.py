
# This code solves for the acoustic field inside domain
# This is just the acoustic field inside a fluid with no solid domain
# It outputs data designed to train a neural operator

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helmholtz2d import helmholtz2d

import numpy as np
from numpy import meshgrid
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from collections import OrderedDict

meshName = "2DElectroElasticAcoustic/Upto10MHz"

fList = [2.4e6, 4.8E6, 7.2E6, 9.6E6]
rhoList = [500, 1000, 1500, 2000]
aList = [5, 10, 15, 20] # acceleration on boundary (m / s^2)

c = 1481 # speed of sound

d_constraints = OrderedDict() # very important this is ordered. This ensures the constraints added last override those added earlier
epsilon = 1e-6

solutions, fVals, rhoVals, aVals = [], [], [], []
XRef, YRef = None, None

for f in fList:
    for rho in rhoList:
        for a in aList:
            def n0_acoust():
                def mask(x, y):
                    mask = np.isclose(x, 0.00015, atol = epsilon)
                    return mask
                def vals(x, y):
                    return a * rho # convert acceleration to normal pressure grad
                return mask, vals

            hm = helmholtz2d(f = f, c = c, rho = rho, domain = [3], meshName = meshName, neumann_constraint_functions = [n0_acoust], builder = 'python') #, reorder = [cm.X, cm.Y])

            if XRef is None:
                XRef, YRef = hm.X, hm.Y
            else:
                assert np.allclose(hm.X, XRef) and np.allclose(hm.Y, YRef), "mesh changed between runs"

            solutions.append(hm.u)
            fVals.append(f)
            rhoVals.append(rho)
            aVals.append(a)

# save mesh, solutions, and associated parameters to file
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'acousticDataset.npz'),
         X = XRef, Y = YRef, U = np.stack(solutions),
         f = np.array(fVals), rho = np.array(rhoVals), a = np.array(aVals), c = c)

