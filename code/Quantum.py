import numpy as np 

def MKron(m_13): 
    m_13 = m_13/np.linalg.norm(m_13)
    m_25 = m_13
    m_46 = m_13
    M1 = np.kron(np.kron(m_13,m_25),m_46)
    M = M1.reshape(6*[2])
    M = np.transpose(M,[0,2,1,4,3,5]).flatten()
    return M


def Infidelity(Phi, Psi):
    Inf = 1 - abs(np.vdot(Phi,Psi))**2
    return Inf


def triple_kron( psi1, psi2, psi3 ):
    return np.kron( np.kron( psi1, psi2 ), psi3)


def  RandomState( dim ):
    psi = np.random.randn(dim) + 1j*np.random.randn(dim)
    psi = psi/np.linalg.norm(psi)
    return psi 


def vector2observable( z ):
    return np.eye(2) - 2*np.outer( z,z.conj() )


def Outer2Kron( A, Dims ):
    # From vec(A) outer vec(B) to A kron B
    N   = len(Dims)
    Dim = A.shape
    A   = np.transpose( A.reshape(2*Dims), np.array([range(N),range(N,2*N) ]).T.flatten() ).flatten()
    return A.reshape(Dim)


def Kron2Outer( A, Dims ):
    # From A kron B to vec(A) outer vec(B)
    N   = len(Dims)
    Dim = A.shape
    A   = np.transpose( A.reshape( np.kron(np.array([1,1]),Dims) ), np.array([range(0,2*N,2),range(1,2*N,2)]).flatten() ).flatten()
    return A.reshape(Dim)
    

def LocalProduct( Psi, Operators , Dims=[] ):
    """
    Calculate the product (A1xA2x...xAn)|psi>
    """
    sz = Psi
    if not Dims: 
        Dims = [ Operators[k].shape[-1] for k in range( len(Operators) ) ]
    N = len(Dims)
    for k in range(N):
        Psi  = (( Operators[k]@Psi.reshape(Dims[k],-1) ).T ).flatten()
    return Psi


def InnerProductMatrices( X, B, Vectorized = False ):
    """
    Calculate the inner product tr( X [B1xB2x...xBn])
    """
    X = np.array(X)
    
    if isinstance(B, list): 
        B = B.copy()
        nsys = len(B)
        nops = []
        Dims = []
        if Vectorized == False :
            for j in range(nsys):
                B[j] = np.array(B[j])
                if B[j].ndim == 2 :
                    B[j] = np.array([B[j]])
                nops.append( B[j].shape[0] )
                Dims.append( B[j].shape[1] )
                B[j] = B[j].reshape(nops[j],Dims[j]**2)
        elif Vectorized == True :
            for j in range(nsys):
                nops.append( B[j].shape[0] )
                Dims.append( int(np.sqrt(B[j].shape[1])) )                
        
        if X.ndim == 2 :       
            TrXB = LocalProduct( Outer2Kron( X.flatten(), Dims ), B ) 
        elif X.ndim == 3 :
            TrXB = []
            for j in range( X.shape[0] ):
                TrXB.append( LocalProduct( Outer2Kron( X[j].flatten(), Dims ), B ) )
        elif X.ndim == 1:
            TrXB = LocalProduct( Outer2Kron( X, Dims ), B ) 
        
        return np.array( TrXB ).reshape(nops).flatten()
        
    elif isinstance(B, np.ndarray):     
        
        if B.ndim == 2 and Vectorized == False :
            return np.trace( X @ B )
        
        elif B.ndim == 4 :
            nsys = B.shape[0]
            nops = nsys*[ B[0].shape[0] ]
            Dims = nsys*[ B[0].shape[1] ]
            B = B.reshape(nsys,nops[0],Dims[0]**2)
            
        elif B.ndim == 3 :
            if Vectorized == False :
                nsys = 1
                nops = B.shape[0]       
                Dims = [ B.shape[1] ]
                B = B.reshape(nsys,nops,Dims[0]**2)
            if Vectorized == True :
                nsys = B.shape[0]
                nops = nsys*[ B[0].shape[0] ]
                Dims = nsys*[ int(np.sqrt(B[0].shape[1])) ]
        if X.ndim == 2 :       
            TrXB = LocalProduct( Outer2Kron( X.flatten(), Dims ), B ) 
        elif X.ndim == 3 :
            TrXB = []
            for j in range( X.shape[0] ):
                TrXB.append( LocalProduct( Outer2Kron( X[j].flatten(), Dims ), B ) )
        elif X.ndim == 1:
            TrXB = LocalProduct( Outer2Kron( X, Dims ), B ) 

        return np.array( TrXB ).reshape(nops).flatten()
    
from itertools import product

def sqr_matrix_combination(c, B):
    """
    Computes:
        Z = sum_{m1,...,mS} c[m1,...,mS] kron(B_{m1},...,B_{mS})

    Parameters
    ----------
    c : ndarray
        S-dimensional coefficient array
    B : list of ndarray
        Each element must be (R, R, M) or (R, C, M)

    Returns
    -------
    Z : ndarray
        Resulting square matrix
    other : dict
        Metadata
    """

    if not isinstance(B, list):
        raise ValueError("B must be a list of arrays (one per subsystem).")

    S = len(B)

    R = []
    C = []
    M = []
    processed = []

    for Bs in B:

        Bs = np.array(Bs)

        if Bs.ndim == 2:
            #MAL
            m, r = Bs.shape
            cdim = 1
            Bs = Bs.reshape(m, r, 1)

        elif Bs.ndim == 3:
            m, r, cdim = Bs.shape

        else:
            raise ValueError("Each subsystem must be 2D or 3D array")

        R.append(r)
        C.append(cdim)
        M.append(m)

        mats = []
        for k in range(m):
            W = Bs[k,:, :]
            if r != cdim:
                W = W @ W.T
            mats.append(W)

        processed.append(mats)

    total_dim = np.prod(R)
    Z = np.zeros((total_dim, total_dim), dtype=complex)

    for indices in product(*[range(m) for m in M]):
        coeff = c[indices]

        kron_term = processed[0][indices[0]]
        for s in range(1, S):
            kron_term = np.kron(kron_term, processed[s][indices[s]])

        Z += coeff * kron_term

    return Z