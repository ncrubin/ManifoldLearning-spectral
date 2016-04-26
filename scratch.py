
def test1():
    s = sdp.SDP()
    s.sdpfile="/Users/nick/sdpsolve/examples/example4/Hub1DN4U0DQG.sdp"
    s.Initialize()
    bpsdp.solve_bpsdp(s)

    return s


def spiral():

    X = np.array([
        [0,1,0],
        [1,0,0],
        [0,-1,0],
        [-1,0,0],
        [0,2,0],
        [2,0,0],
        [0,-2,0],
        [-2,0,0]
        ])

    return X

def test_roll(X):

    distances = np.zeros((X.shape[0]-1, 2))
    indices = np.zeros((X.shape[0]-1,2),dtype=int)

    for i in range(X.shape[0]-1):

        distances[i,0] = 0.0
        distances[i,1] = np.linalg.norm(X[i,:] - X[i+1,:])
        indices[i,0] = i
        indices[i,1] = i+1

    return distances, indices

def test_spectraL_basic():

    #Generate Swiss Roll data
    #np.random.seed(1)
    k = 2
    samples = 8
    X, color = datasets.samples_generator.make_swiss_roll(n_samples=samples,
            random_state=2)
    X = spiral()
    Xp = np.copy(X)

    #Xp = np.insert(X,[3],np.random.normal(loc=0.0,scale=0.44721359549995793,size=(X.shape[0],3)),axis=1)
    #project_axis = np.array([[1,0,0],[0,0,0],[0,0,1]])
    #Xp = np.dot(Xp,project_axis)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Xp[:,0], Xp[:,1], Xp[:,2], c=color, cmap=plt.cm.Spectral)
    #plt.show()
    #sys.exit()

    #check points are centered
    print "X center ", np.sum(Xp[:,0])/Xp.shape[0]
    print "Y center ", np.sum(Xp[:,1])/Xp.shape[0]
    print "Z center ", np.sum(Xp[:,2])/Xp.shape[0]

    Xp = Xp - np.array([ np.sum(Xp[:,0])/Xp.shape[0],
        np.sum(Xp[:,1])/Xp.shape[0] ,np.sum(Xp[:,2])/Xp.shape[0] ])

    print "X center ", np.sum(Xp[:,0])/Xp.shape[0]
    print "Y center ", np.sum(Xp[:,1])/Xp.shape[0]
    print "Z center ", np.sum(Xp[:,2])/Xp.shape[0]


    #############
    #do unrolling
    #############

    #primal variable
    K = np.zeros((Xp.shape[0],Xp.shape[0]))
    ndim = Xp.shape[0]

    #calculate distance matrix of inputs
    #nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(Xp)
    #distances, indices = nbrs.kneighbors(Xp)

    distances, indices = test_roll(Xp)



    print "primal Bound ", ((200**3)*np.max(distances)**2)/2
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(Xp[:,0], Xp[:,1], Xp[:,2], c=color, cmap=plt.cm.Spectral)
    #for i in range(indices.shape[0]):
    #    for j in range(1,k):
    #        ax.plot([Xp[i,0], Xp[indices[i,j],0]], [Xp[i,1], Xp[indices[i,j],1]],
    #        [Xp[i,2], Xp[indices[i,j],2]] )
    #plt.show()
    #sys.exit()



    #set up SDP
    s = sdp.SDP()
    s.Initialize()
    s.nv = ndim**2
    s.blockstruct = [ndim]
    s.nb = 1

    #initialize cvec, bvec
    s.cvec = np.require(-1*np.eye(K.shape[0]).reshape((s.nv,1)), dtype=float,
    requirements=['F','A','W','O'])
    s.bvec = []

    cnstr_num = 0
    #centered constraint
    Amatrow = [cnstr_num]*(Xp.shape[0]**2)
    Amatcol = range(Xp.shape[0]**2)
    Amatarc = [1]*(Xp.shape[0]**2)
    s.bvec.append(0.0)
    cnstr_num += 1

    #local isometry constraint
    for i in range(indices.shape[0]):
        for j in range(k):
            print i, indices[i,j], i < indices[i,j]
            if i < indices[i,j]:

                Amatrow += [cnstr_num]*4
                Amatcol += [i*ndim + i, i*ndim + indices[i,j],
                    indices[i,j]*ndim + i, indices[i,j]*ndim + indices[i,j] ]
                Amatarc += [1,-1,-1,1]
                s.bvec.append(distances[i,j])
                print Amatrow[-4:]
                print Amatcol[-4:]
                print Amatarc[-4:]
                cnstr_num += 1

    s.nnz = len(Amatarc)
    s.nc = cnstr_num
    s.bvec = np.require(np.array(s.bvec), dtype=float,
    requirements=['F','A','W','O'])
    s.bvec = s.bvec.reshape((s.bvec.shape[0],1))
    s.iter_max = 100000
    s.inner_solve = "EXACT"
    #print np.max(Amatrow), s.nc, s.bvec.shape, s.nnz, len(Amatrow), len(Amatarc), len(Amatcol)
    s.Amat = csc_matrix((Amatarc,(Amatrow,Amatcol)), shape=( s.nc, s.nv ) )

    for i in range(1,8):
        print s.Amat.getrow(i)
        print s.bvec[i]


    #writeSDP(s, Amatrow, Amatcol, Amatarc)
    #sys.exit()

    #bpsdp.solve_bpsdp_max(s)
    #bpsdp.solve_bpsdp(s)
    rrsdp.solve_rrsdp(s)


    Rmat = s.primal[0].reshape((ndim,ndim))

    w, v = np.linalg.eigh(np.dot(Rmat,Rmat.T))

    #find dimension of lower sub space
    wdiff = []
    for i in range(1,w.shape[0]):
        wdiff.append(w[i] - w[i-1])

    print np.argmax(wdiff)

    d = w.shape[0] - (np.argmax(wdiff) + 1)

    Y = np.dot(np.diag(np.sqrt(w[-d:])),v[:,-d:].T)

    print Y.shape
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Y, np.zeros(Y.shape), np.zeros(Y.shape), c=color, cmap=plt.cm.Spectral)
    plt.show()
