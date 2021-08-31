import utils
import math
from torch.distributions import Categorical
import torch

class GenericExactState:
    '''
    Represents a generic state, as a table of complex coefficients in the
    computational basis. Use only up to 12 qubits or so
    '''
    def __init__(self, state):
        '''
        state: 2**n dimensional Complex tensor. Assume normalized.
        Interpreted as a ket.
        '''
        self.state = state
        self.num_sites = int(math.log2(state.shape[0])+.5)
        assert 2**self.num_sites == state.shape[0]

    def to(self, device):
        '''
        Saves the self.state attribute in the memory of a specified device.
        Uses the torch.to(device) functionality to switch between
        a GPU to CPU, for example

        device: torch.device()
        '''
        self.state = self.state.to(device)
        return self

    def fidelity_to(self, other):
        '''
        other: GenericExactState
        '''
        overlap = utils.Complex.einsum('i,i', self.state.conjugate(),
                                       other.state.to(self.state.real.dtype
                                       ).to(self.state.real.device))

        fidelity = overlap.norm2()
        return fidelity

    def overlap_to(self, other):
        """
        <self| other>
        Args:
            other:

        Returns:

        """
        overlap = utils.Complex.einsum('i,i', self.state.conjugate(),
                                       other.state.to(self.state.real.dtype
                                       ).to(self.state.real.device))
        return overlap

    def amplitudes(self, samples):
        samples_long = utils.bits_to_long(samples)
        return self.state[samples_long]

    def full_state(self):
        return self.state

    def amplitudes_in_basis(self, samples, basis):
        contributing, overlaps = utils.basis_change(samples, basis)
        amplitudes_compbasis = self.amplitudes(contributing)
        overlaps = overlaps.to(self.state.real.dtype)
        amplitudes = utils.Complex.einsum('bi,bi->b', overlaps, amplitudes_compbasis)
        return amplitudes

    def all_amplitudes_in_basis(self, basis):
        contributing, contributions = utils.construct_sparse_unitary(basis)
        contributions = contributions.to(self.state.real.dtype)
        return (self.state[contributing]*contributions).sum(dim=-1)

    def sample_in_basis(self, basis, num_samples, return_logprobs = False):
        amps = self.all_amplitudes_in_basis(basis)
        probs = amps.norm2()
        dist = Categorical(probs=probs)

        samples_long = dist.sample((num_samples,))
        samples = utils.long_to_bits(samples_long, self.num_sites)
        if return_logprobs:
            logprobs = dist.log_prob(samples_long)
            return samples, logprobs
        else:
            return samples

    def sample(self, num_samples, return_logprobs = False):
        probs = self.state.norm2()
        assert (probs.sum()-1).abs() < 1e-6
        dist = Categorical(probs=probs)

        samples_long = dist.sample((num_samples,))
        samples = utils.long_to_bits(samples_long, self.num_sites)
        if return_logprobs:
            logprobs = dist.log_prob(samples_long)
            return samples, logprobs
        return samples

    def renyi2_ee(self, partition):
        """
        Second Renyi entropy of the reduced DM, obtained by tracing over
        the first `partition` sites
        """
        density_matrix = utils.Complex.einsum(  'i,j->ij', 
                                                self.state,
                                                self.state.conjugate())
        for _ in range(partition):
            half = density_matrix.shape[0]//2
            density_matrix = density_matrix[:half, :half] + density_matrix[half:, half:]
        trrho2 = utils.Complex.einsum('ij,ji', density_matrix, density_matrix)
        return - trrho2.real.log2()
