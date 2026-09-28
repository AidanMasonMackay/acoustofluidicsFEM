
import matplotlib.pyplot as plt

import numpy as np
import os

cwd = os.path.dirname(os.path.abspath(__file__))

file = os.path.join(cwd, "acousticDataset.npz")
data = np.load(file)

testID = 0

X = data["X"]
Y = data["Y"]
U = data["U"][testID]

a = data["a"][testID]
rho = data["rho"][testID]
f = data["f"][testID]


fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(projection="3d")
sc = ax.scatter(X, Y, U, c=U, cmap="hsv")
plt.title(f"Acoustic field (Pa), f = {f*10**6}MHz, rho = {rho}, a = {a}")
plt.colorbar(sc)
plt.show()

