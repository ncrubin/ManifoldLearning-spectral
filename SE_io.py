def writeSDP(s, Amatrow, Amatcol, Amatarc ):

    with open("SDPSpectralLearn.sdp",'w') as fid:

        fid.write("%i\t%i\t%i\t%i\n"%(s.nc, s.nv, s.nnz, s.nb))
        fid.write("%i\n"%(s.blockstruct[0]))

        #write amatrix
        for i in range(len(Amatrow)):
            fid.write("%i\t%i\t%4.10E\n"%(Amatrow[i]+1, Amatcol[i]+1, Amatarc[i]))

        #write bmatrix
        for i in range(s.bvec.shape[0]):
            fid.write("%4.10E\n"%(s.bvec[i]))


        #write cmatrix
        for i in range(s.cvec.shape[0]):
            fid.write("%4.10E\n"%(s.cvec[i]))
