import numpy as np
from codes.Quantum import InnerProductMatrices, sqr_matrix_combination

import numpy as np

def project_simplex(v):
    """
    Projection of vector v onto probability simplex:
    v_i >= 0 and sum v_i = 1
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)

    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)

    w = np.maximum(v - theta, 0)
    return w


def project_to_physical_density_matrix(rho):
    """
    Projection to closest positive semidefinite trace-1 matrix
    following Smolin-Gambetta-Smith (2011).
    """

    # ensure Hermitian
    rho = (rho + rho.conj().T) / 2

    # eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(rho)

    # project eigenvalues to simplex
    eigvals_proj = project_simplex(eigvals)

    # reconstruct matrix
    rho_proj = eigvecs @ np.diag(eigvals_proj) @ eigvecs.conj().T

    return rho_proj


class EnchancedShadowTomography:

    def __init__(self, n, N ):
        self.n = n
        self.N = N
        # Matrices de Pauli
        sigma_x = (1/np.sqrt(2))*np.array([[1, 1],
                                        [1, -1]])
        sigma_y = (1/np.sqrt(2))*np.array([[1, 1],
                                        [1j, -1j]]).T.conj()
        sigma_z = np.array([[1, 0],
                            [0, 1]])
        id = np.array(np.eye(2))
        self.id = id 
        self.VSigma = [ sigma_x, sigma_y, sigma_z ]

        x0 = np.array([[1,1],[1,1]])/2
        x1 = np.array([[1,-1],[-1,1]])/2
        y0 = np.array([[1,-1j],[1j,1]])/2
        y1 = np.array([[1,1j],[-1j,1]])/2
        z0 = np.array([[1,0],[0,0]])
        z1 = np.array([[0,0],[0,1]])

        self.proj = [ x0, x1, y0, y1, z0, z1 ]

        self.inv  = [ 3*x0-id, 3*x1-id, 
                        3*y0-id, 3*y1-id,
                        3*z0-id, 3*z1-id, ]

    def train_shadows(self, rho ):
        probs = InnerProductMatrices( rho, self.n*[self.proj] )/3
        probs = np.abs(probs)
        probs = probs / np.sum(probs)
        shots = np.random.multinomial( self.N, probs ) 
        self.shots_per_outcome = shots    
        shots = shots.reshape(self.n*[3,2])
        self.shots_per_pauli = np.sum( shots, tuple(np.arange(1,2*self.n,2)) )

    def enhance_shadow( self, sigma ):
        self.sigma = sigma 
        probs = InnerProductMatrices( sigma, self.n*[self.proj] ).reshape(self.n*[3,2])
        shots_per_pauli = self.shots_per_pauli.reshape(self.n*[3,1])
        shots_per_outcome = probs * shots_per_pauli
        self.shots_per_outcome_sigma = shots_per_outcome.reshape(-1) 

    def estimate_global_observable( self, A ):
        A_shot = InnerProductMatrices( A, self.n*[self.inv] )
        A_mean = np.dot( self.shots_per_outcome, A_shot ) / self.N 
        return np.real( A_mean ) 
    
    def estimate_global_observable_enhanced( self, A ):
        A_shot = InnerProductMatrices( A, self.n*[self.inv] )
        A_rho = np.dot( self.shots_per_outcome, A_shot ) / self.N 
        A_sigma = np.dot( self.shots_per_outcome_sigma, A_shot ) / self.N 
        return np.real( A_rho -  A_sigma + np.trace( A@self.sigma ) )
    
    def estimate_density_matrix( self ):
        rho_hat = sqr_matrix_combination( self.shots_per_outcome.reshape(self.n*[6])/self.N, self.n*[self.inv] )
        return project_to_physical_density_matrix( rho_hat ) 

class ShadowTomography:

    def __init__(self, n, N ):
        self.n = n
        self.N = N
        # Matrices de Pauli
        sigma_x = (1/np.sqrt(2))*np.array([[1, 1],
                                        [1, -1]])
        sigma_y = (1/np.sqrt(2))*np.array([[1, 1],
                                        [1j, -1j]]).T.conj()
        sigma_z = np.array([[1, 0],
                            [0, 1]])
        id = np.array(np.eye(2))
        self.VSigma = [id, sigma_x, sigma_y, sigma_z]

    def random_unitaries(self):
        U = []
        U_local = []
        for i in range(self.N):
            k = np.random.randint(1,4, size = self.n)
            #print(k)
            U_i = 1
            U_local_i = []
            for j in k:
                U_i = np.kron(U_i,self.VSigma[j])
                U_local_i.append(self.VSigma[j])
            U.append(U_i)  #Matrices aleatorias en kron
            U_local.append(U_local_i)
        self.U = U
        self.U_local = U_local 

    def estimate_shadows( self, rho ):
        b = []
        for U_i in self.U:
            Prob = np.abs(np.diag(U_i@rho@U_i.conj().T))
            Prob = Prob/np.sum(Prob)
            b.append(np.random.multinomial(1,Prob).argmax())
        self.b = b

    def estimate_state(self):
        Rhos = 0
        for j in range(self.N):
            U_local_j = self.U_local[j]
            b_j = bin(self.b[j])[2:].zfill(self.n)
            rhos = 1
            for i in range(self.n):
                u = U_local_j[i].T.conj()
                u = u[:,int(b_j[i])]
                rhos = np.kron(rhos, 
                                3*np.outer(u,u.conj())-np.eye(2))
            Rhos += rhos
        return Rhos/self.N 
        
    def train_shadows(self, rho ):
        self.random_unitaries()
        self.estimate_shadows(rho)

    # def estimate_observable(self, A ):
    #     Ashadows = []
    #     for i in range(self.N):
    #         Ashadows.append(np.trace(self.Rhos[i]@A))
    #     return np.mean(Ashadows)
    
    def estimate_observable(self, A_local):
        """
        A_local: lista de matrices 2x2 (Pauli string)
        """
        vals = []
        for j in range(self.N):
            U_local_j = self.U_local[j]
            b_j = bin(self.b[j])[2:].zfill(self.n)
            val = 1.0
            for i in range(self.n):
                u = U_local_j[i].T.conj()[:, int(b_j[i])]
                # 3<u|P|u> - Tr(P)
                local_val = 3 * (u.conj().T @ A_local[i] @ u) \
                            - np.trace(A_local[i])
                val *= local_val
            vals.append(val)
        return np.real(np.mean(vals))

    def estimate_fidelity(self, psi):
        """
        Estima <psi|rho|psi> correctamente para local 2-design.
        Usa solo U_local y no construye matrices 2^n x 2^n.
        """
        vals = []
        n = self.n
        dim = 2**n
        for j in range(self.N):
            b_j = bin(self.b[j])[2:].zfill(n)
            U_local_j = self.U_local[j]
            phi = psi.copy()
            # aplicar operador local O_k = 3|v><v| - I
            for k in range(n):
                v = U_local_j[k].conj().T[:, int(b_j[k])]
                O_k = 3*np.outer(v, v.conj()) - np.eye(2)
                # aplicar sobre qubit k
                phi = phi.reshape([2]*n)
                phi = np.moveaxis(phi, k, 0)
                phi = np.tensordot(O_k, phi, axes=([1],[0]))
                phi = np.moveaxis(phi, 0, k)
                phi = phi.reshape(dim)
            vals.append(np.vdot(psi, phi).real)
        fid = np.mean(vals)
        # if fid < 0:
        #     fid=0
        # elif fid>1:
        #     fid=1
        return fid
    
    def estimate_observable_pair(self, Pi_pairs):
        """
        Estima Tr( rho (Π_{1,2} ⊗ Π_{3,4} ⊗ ...) )
        
        Pi_pairs: lista de matrices 4x4 (una por par de qubits)
        """
        n = self.n
        assert n % 2 == 0
        num_pairs = n // 2
        vals = []
        for j in range(self.N):
            b_j = bin(self.b[j])[2:].zfill(n)
            U_local_j = self.U_local[j]
            total_val = 1.0
            for m in range(num_pairs):
                q1 = 2*m
                q2 = 2*m + 1
                # vectores locales v = U†|b>
                v1 = U_local_j[q1].conj().T[:, int(b_j[q1])]
                v2 = U_local_j[q2].conj().T[:, int(b_j[q2])]
                P1 = np.outer(v1, v1.conj())
                P2 = np.outer(v2, v2.conj())
                Pi = Pi_pairs[m]
                if Pi.ndim==1:
                    Pi = np.outer( Pi, Pi.conj() )
                # términos de la expansión
                term1 = 9 * np.trace(
                    np.kron(P1, P2) @ Pi
                )
                term2 = -3 * np.trace(
                    np.kron(P1, np.eye(2)) @ Pi
                )
                term3 = -3 * np.trace(
                    np.kron(np.eye(2), P2) @ Pi
                )
                term4 = np.trace(Pi)
                block_val = (term1 + term2 + term3 + term4).real
                total_val *= block_val
            vals.append(total_val)
        return np.mean(vals)

    def estimate_fidelity_pairs(self, psi_pairs):

        n = self.n
        num_pairs = n // 2
        vals = []
        for j in range(self.N):
            b_j = bin(self.b[j])[2:].zfill(n)
            U_local_j = self.U_local[j]
            total_val = 1.0
            for m in range(num_pairs):
                q1 = 2*m
                q2 = 2*m + 1
                v1 = U_local_j[q1].conj().T[:, int(b_j[q1])]
                v2 = U_local_j[q2].conj().T[:, int(b_j[q2])]
                psi = psi_pairs[m].reshape(2,2)
                # término 9 |<psi|v1⊗v2>|²
                amp = np.tensordot(
                    psi.conj(),
                    np.outer(v1, v2),
                    axes=([0,1],[0,1])
                )
                term1 = 9 * np.abs(amp)**2
                # término -3 <psi|P1⊗I|psi>
                term2 = -3 * np.vdot(
                    psi,
                    np.tensordot(
                        np.outer(v1,v1.conj()),
                        psi,
                        axes=([1],[0])
                    )
                ).real
                # término -3 <psi|I⊗P2|psi>
                term3 = -3 * np.vdot(
                    psi,
                    np.tensordot(
                        psi,
                        np.outer(v2,v2.conj()),
                        axes=([1],[0])
                    )
                ).real
                block_val = term1 + term2 + term3 + 1
                total_val *= block_val
            vals.append(total_val)
        return np.mean(vals)

    
def proyectar_al_simplex(v):
    """
    Encuentra la distribución de probabilidad p más cercana al vector v
    en términos de distancia Euclídea.
    """
    v = np.array(v, dtype=float)
    n_features = len(v)
    # 1. Ordenar el vector de forma descendente
    u = np.sort(v)[::-1]
    # 2. Calcular la suma acumulada
    cssv = np.cumsum(u)
    # 3. Identificar el valor rho (el número de componentes que serán mayores a 0)
    ind = np.arange(1, n_features + 1)
    cond = u - (cssv - 1) / ind > 0
    rho = ind[cond][-1]
    # 4. Calcular el multiplicador de Lagrange (theta)
    theta = (cssv[rho - 1] - 1) / rho
    # 5. Calcular la proyección final
    p = np.maximum(v - theta, 0)
    return p