def KNN_NCR(X, k):
    """
    Brute force algorithm for finding k-nearest-neighbors given a data set.

    input 1) X matrix where rows are data-points
    input 2) k integer.  number of nearest-neighbors.

    return 1) indices of k-nearest-neighbors for each row of X
    return 2) distances of k-nearest-neighbors for each row of X

    """

    N = X.shape[0]
    indices = np.zeros((N,k),dtype=int) #indices of nearest-neighbors
    dist = np.zeros((N,k),dtype=float) #distances of nearest-neighbors
    for i in range(N):
        target = X[i,:]
        q = [] #k-nearest X[j,:] stored as a tuple and sorted
        for j in range(N):

            if i!=j:

                tdist = np.linalg.norm(X[i,:]-X[j,:])

                if len(q) < k:

                    q.append((j,tdist,X[j,:]))

                else:

                    if q[-1][1] > tdist:

                        q.pop(-1)
                        q.append((j,tdist,X[j,:]))

                q = sorted(q, key=lambda y: y[1])

        indices[i,:] = np.array(map(lambda y: y[0],q))
        dist[i,:] = np.array(map(lambda y: y[1],q))

    return indices, dist
