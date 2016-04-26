#system imports
import os
import sys
sys.path.append("/Users/nick") #adds path to sdpsolve

#SDP solver inputs
from sdpsolve.sdp import sdp
from sdpsolve.bpsdp import bpsdp
from sdpsolve.rrsdp import rrsdp
from sdpsolve.csdp import csdp
from SwissRoll import make_swiss


#sklearn inputs for datasets
from sklearn import manifold, datasets
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.sparse import csc_matrix

DEBUG = True

def GenerateRollData(k=8, samples=800, visualize=False, center=False):

    np.random.seed(1)
    X, color = datasets.samples_generator.make_swiss_roll(n_samples=samples,
            random_state=10)

    #X, color = datasets.make_s_curve(n_samples=samples, noise=0.0, random_state=None)
    Xp = np.copy(X)
    Xp[:,1] *= 0.2

    #add up to dim=6
    #Xp = np.insert(X,[3],np.random.normal(loc=0.0,scale=0.44721359549995793,size=(X.shape[0],3)),axis=1)
    #project_axis = np.array([[1,0,0],[0,0,0],[0,0,1]])
    #Xp = np.dot(Xp,project_axis)

    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(Xp[:,0], Xp[:,1], Xp[:,2], c=color, cmap=plt.cm.Spectral)
        plt.show()

    if center:

        ##check points are centered
        print("point centers")
        for dim in range(Xp.shape[1]):
            print "dim %i center ", np.sum(Xp[:,dim])/Xp.shape[0]

        center = map(lambda x: np.sum(Xp[:,x])/Xp.shape[0], range(Xp.shape[1]))
        center = np.array(center)
        #center points
        Xp = Xp - center

        print("recentered")
        for dim in range(Xp.shape[1]):
            print "dim %i center ", np.sum(Xp[:,dim])/Xp.shape[0]

    return Xp, color


def writeBPSDP():

    s = sdp.SDP()
    s.Initialize()
    s.nv = ndim**2
    s.blockstruct = [ndim]
    s.nb = 1

    #initialize cvec, bvec
    s.cvec = np.require(np.eye(K.shape[0]).reshape((s.nv,1)), dtype=float,
    requirements=['F','A','W','O'])
    s.bvec = []

    #if we are using CSDP then pass only the upper triangle of C and cosntraints
    cnstr_num = 0
    Amatrow = [cnstr_num]*(Xp.shape[0]**2)
    Amatcol = range(Xp.shape[0]**2)
    Amatarc = [1]*(Xp.shape[0]**2)
    s.bvec.append(0.0)
    cnstr_num += 1

    #local isometry constraint
    for i in range(indices.shape[0]):
        for j in range(k):

            if i < indices[i,j]:

                Amatrow += [cnstr_num]*4
                Amatcol += [i*ndim + i, i*ndim + indices[i,j],
                     ndim*indices[i,j] + i, indices[i,j]*ndim + indices[i,j] ]
                Amatarc += [1,-1,-1, 1]
                xi = Xp[i,:]
                xj = Xp[indices[i,j],:]
                s.bvec.append(np.dot(xi,xi) + np.dot(xj,xj) - 2*np.dot(xi,xj))
                #s.bvec.append(distances[i,j])

                cnstr_num += 1

    s.nnz = len(Amatarc)
    s.nc = cnstr_num
    s.bvec = np.require(np.array(s.bvec), dtype=float,
    requirements=['F','A','W','O'])
    s.bvec = s.bvec.reshape((s.bvec.shape[0],1))
    s.iter_max = 100
    s.inner_solve = "CG"
    s.Amat = csc_matrix((Amatarc,(Amatrow,Amatcol)), shape=( s.nc, s.nv ) )

    return s

def writeCSDPfile(ndim, indices, distances, nbrs  ):

    K = np.zeros((ndim, ndim))
    s = sdp.SDP()
    s.Initialize()
    s.nv = ndim**2
    s.blockstruct = [ndim]
    s.nb = 1

    #initialize cvec, bvec
    s.cvec = np.require(np.eye(K.shape[0]).reshape((s.nv,1)), dtype=float,
    requirements=['F','A','W','O'])
    s.bvec = []

    #if we are using CSDP then pass only the upper triangle of C and cosntraints

    cnstr_num = 0
    dim = Xp.shape[0]
    squaredim = dim*(dim + 1)//2
    Amatrow = [cnstr_num]*(squaredim)
    #get upper triangle indices from numpy function
    pairsIndices = np.triu_indices(dim)
    Amatcol = map(lambda x: x[0]*dim + x[1], zip(pairsIndices[0], pairsIndices[1]) )
    Amatarc = [1]*(squaredim)
    s.bvec.append(0.0)
    cnstr_num += 1


    #local isometry constraint
    for i in range(indices.shape[0]):
        for j in range(k):

            if i < indices[i,j]:

                Amatrow += [cnstr_num]*3
                Amatcol += [i*ndim + i, i*ndim + indices[i,j],
                    indices[i,j]*ndim + indices[i,j] ]
                Amatarc += [1,-1,1]
                xi = Xp[i,:]
                xj = Xp[indices[i,j],:]
                s.bvec.append(np.dot(xi,xi) + np.dot(xj,xj) - 2*np.dot(xi,xj))
                #s.bvec.append(distances[i,j])

                cnstr_num += 1

    s.nnz = len(Amatarc)
    s.nc = cnstr_num
    s.bvec = np.require(np.array(s.bvec), dtype=float,
    requirements=['F','A','W','O'])
    s.bvec = s.bvec.reshape((s.bvec.shape[0],1))
    s.iter_max = 100
    s.inner_solve = "CG"
    s.Amat = csc_matrix((Amatarc,(Amatrow,Amatcol)), shape=( s.nc, s.nv ) )

    return s

def SemiDefiniteEmbedding(Xp, k, color=None, d=None):
    """
    Peform semidefinite embedding non-linear dimensionality reduction on input
    dataset `X` with `k` connected nearest-neighbors enforcing k-local isometry.
    output will be the new data set with reduced dimension `Y` that has been
    projected down to have dimensionality `d`. Method based on Semidefinite embedding
    described in [link](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.221.8829&rep=rep1&type=pdf#page=390)
    by Kilian Q. Weinberger, Benjamin D. Packer, and Lawrence K. Saul.

        input:
            X = (m, n) matrix of m samples with n features each
            k = dimension of local nearest-neighbors
            optional= 'd' dimension desired in output data.  Use if eigenvalues
            of kernel don't have a clear divide.

        return:
            Y = new dataset rotated down to dimension d
            d = dimension of new dataset
    """

    #################################################
    #
    # Set up Kernel and plot neighbor connections
    #
    ##################################################

    #Kernel initialization
    ndim = Xp.shape[0]

    #calculate distance matrix of inputs
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(Xp)
    distances, indices = nbrs.kneighbors(Xp)

    if DEBUG and color != None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(Xp[:,0], Xp[:,1], Xp[:,2], c=color, cmap=plt.cm.Spectral)

        for i in range(indices.shape[0]):
            for j in range(1,k):

                ax.plot( [Xp[i,0], Xp[indices[i,j],0]] , [Xp[i,1], Xp[indices[i,j],1]],
                [Xp[i,2], Xp[indices[i,j],2]],)

        plt.show()


    #################################################
    #
    # Set up SDP for Kernel
    #
    ##################################################
    s = writeCSDPfile(ndim, indices, distances, nbrs)
    #s = writeBPSDP()

    ########################################
    #
    #  Solve SDP with some method
    #
    ########################################

    #bpsdp.solve_bpsdp_max(s)
    #rrsdp.solve_rrsdp(s)
    #bpsdp.solve_bpsdp(s)
    results = csdp.solve_csdp(s)

    ########################################
    #
    #  Extract Results
    #
    ########################################

    print s.primal.shape, ndim
    Rmat = s.primal.reshape((ndim,ndim))
    w, v = np.linalg.eigh(Rmat)

    #find dimension of lower sub space
    wdiff = []
    for i in range(1,w.shape[0]):
        wdiff.append(w[i] - w[i-1])

    #plt.plot(range(len(wdiff)), wdiff, 'bo-')
    plt.plot(range(w.shape[0]), w, 'bo-')

    plt.show()
    print wdiff
    #d = 2
    #d = w.shape[0] - (np.argmax(wdiff) + 1)

    print "d-dimension ", d
    Y = np.dot(np.diag(np.sqrt(w[-d:])),v[:,-d:].T)

    return Y, d


if __name__=="__main__":



    #generate swiss roll Data
    k = 8; samples = 300; SDPSovler = "CSDP"
    Xp, color = GenerateRollData(k=k, samples=samples, visualize=DEBUG, center=True)

    Y, d = SemiDefiniteEmbedding(Xp, k, color=color, d=2)
    ########################################
    #
    #  Plot results and compare to PCA
    #
    ########################################

    print "Sol shape ", Y.shape
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Y[0,:], Y[1,:], np.zeros_like(Y[0,:]), c=color, cmap=plt.cm.Spectral)
    plt.show()

    print "comparing to PCA"
    Cov = np.dot(Xp.T, Xp)
    wpca, vpca = np.linalg.eigh(Cov)
    print wpca
    Ypca = np.dot(Xp, vpca[:,-2:])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Ypca[:,0], Ypca[:,1], np.zeros(samples), c=color, cmap=plt.cm.Spectral)
    plt.show()
