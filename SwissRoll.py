import numpy as np
import scipy as sp
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def make_swiss(n_samples) :

    n_features =  3
    n_turns, radius = 2., 2.0
    rng = np.random.RandomState(20)
    t = rng.uniform(low=0, high=1, size=n_samples)
    data = np.zeros((n_samples, n_features))
   
    #nick's data
    # generate the 2D spiral data driven by a 1d parameter t
    max_rot = n_turns * 2 * np.pi
    data[:, 0] = radius = t * np.cos(t * max_rot)
    data[:, 1] = radius = t * np.sin(t * max_rot)
    data[:, 2] = rng.uniform(-1, 1.0, n_samples)
    manifold = np.vstack((t * 2 - 1, data[:, 2])).T.copy()
    colors = manifold[:, 0]
 

    #constant * theta

    #####################
    #
    # plotting for debug
    #
    #####################
    #Xp = data
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(Xp[:,0], Xp[:,1], Xp[:,2], c=colors, cmap=plt.cm.Spectral)
    #plt.show()

    return data, colors


if __name__=="__main__":

    make_swiss(400)
