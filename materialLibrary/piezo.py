
import numpy as np

# Material parameters for piezoelectic plate

e0 = 8.85418782*10**-12 # permitivity of free space

c11 = 1.20346e+011
c12 = 7.51791e+010
c13 = 7.50901e+010
c22 = 1.20346e+011
c23 = 7.50901e+010
c33 = 1.10867e+011
c66 = 2.25734e+010

c11_hat = c11 - ((c13**2)/c33)
c12_hat = c12 - ((c13*c23)/c33)
c22_hat = c22 - ((c23**2)/c33)
c66_hat = c66


e31 = -5.35116
e32 = -5.35116
e33 = 15.7835

e31_hat = e31 - ((c12*e33)/c33)
e32_hat = e32 - ((c23*e33)/c33)

rho = 7750 # density
eps = e0*np.array([[1, 0], [0, 1]]) # permitivity matrix
C = np.array([[c11_hat, c12_hat, 0], [c12_hat, c22_hat, 0], [0, 0, c66_hat]]) # elasticity matrix
coupling = np.array([[0, 0], [0, 0], [e31_hat, e32_hat]]) # coupling matrix
