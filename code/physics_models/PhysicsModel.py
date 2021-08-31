import torch
import utils
import scipy.sparse, scipy.sparse.linalg
import exact_solvers

class PhysicsModel():
    def __init__(self):
        """
        Must set self.num_sites
        """
        pass

    def get_hamiltonian_lines(self, *args):
        """
        Given a batch of samples s, returns those samples s' for which <s|H|s'>
        is nonzero, along with the values of <s|H|s'>.
        Args:
            samples: torch uint tensor of shape (num_samples, num_sites)

        Returns:
            a tuple (contributing, contributions)
            contributing: uint tensor of shape (num_samples, num_nonzero, num_qubits)
                containing for each sample the list of basis elements s' for
                which <s|H|s'> is nonzero
                
            contributions: Complex of shape (num_samples, num_nonzero)
                containing <s|H|s'>
        """
        raise NotImplementedError

    def get_hamiltonian_matrix(self):
        """
        return the 2**n x 2**n Hamiltonian Complex matrix
        """
        H = torch.zeros((2 ** self.num_sites, 2 ** self.num_sites), dtype = torch.float)
        H = utils.Complex(H)
        for row in range(2 ** self.num_sites):
            row_state = utils.long_to_bits(torch.tensor([row]), self.num_sites)
            contributing, contributions = self.get_hamiltonian_lines(row_state)
            lines = utils.bits_to_long(contributing[0, :, :])
            H.real[lines, row], H.imag[lines, row] = contributions.real, contributions.imag
        return H

    def exact_expected_energy(self, state):
        """
        Args:
            state (GenericExactState)

        Returns: <psi | H | psi>

        """

        #get Hamiltonian in sparse format
        all_states = utils.all_cb_states(self.num_sites)
        contributing, contributions = self.get_hamiltonian_lines(all_states)
        contributing = utils.bits_to_long(contributing)

        # Calculate Hpsi sparsely:
        Hpsi = (state.state[contributing]*contributions).sum(dim=1)

        psiHpsi = (state.state.conjugate()*Hpsi).sum(dim=0)
        return psiHpsi

    def get_ED_groundstate(self):
        """

        Returns: GenericExactState

        """
        solver = exact_solvers.ExactDiagonalizer(self.get_hamiltonian_matrix())
        solver.solve()
        return solver.get_groundstate()
