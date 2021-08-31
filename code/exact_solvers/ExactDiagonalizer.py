from exact_solvers.GenericExactState import GenericExactState
import numpy as np

import utils

class ExactDiagonalizer:
    '''
    Diagonalizes a Hamiltonian, using np.linalg.eigh. Need to call solve,
    __init__ just saves the Hamiltonian. Use only up to 12 or 13 qubits.
    '''
    def __init__(self, hamiltonian):
        '''
        hamiltonian: 2^n x 2^n Complex tensor, as return
        by physicsmodel.get_hamiltonian_matrix()
        '''
        self.hamiltonian = hamiltonian
        self.hilbertdim = hamiltonian.shape[0]
        assert self.hilbertdim <= 2**13
        self.eigenvals, self.eigenvecs = None, None

    def solve(self, np_type=None):
        '''
        Run the diagonalization, storing the results in self.eigenvals, self.eigenvecs.
        eigenvals are ascending, eigenvecs are normalized. Note that eigenvecs[:, i]
        is the ith eigenvector
        '''
        hamiltonian_np = self.hamiltonian.real.cpu().detach().numpy() \
                                + self.hamiltonian.imag.cpu().detach().numpy() * 1j
        if np_type is not None:
            hamiltonian_np = hamiltonian_np.astype(np_type)
        # eigenvals are ascending, eigenvecs normalized.
        self.eigenvals, self.eigenvecs = np.linalg.eigh(hamiltonian_np)

    def get_groundstate(self):
        '''
        returns an GenericExactState containing the groundstate
        '''
        state = utils.Complex(self.eigenvecs[:, 0])
        return GenericExactState(state)
