import torch
import utils
from exact_solvers import GenericExactState


class NeuralQuantumState(torch.nn.Module):
    """
    Parent class for NQSs, representing pure qubit wavefunctions by representing
    their amplitudes in the computational basis.

    The methods amplitudes, all_amplitudes, amplitudes_in_basis
    and all_amplitudes_in_basis are meant for external use.
    """
    def __init__(self):
        super(NeuralQuantumState, self).__init__()

    def forward(self, *args):
        raise NotImplementedError

    def sample(self, num_samples):
        """
        Samples computational basis states |s> according to |<s|psi>|^2.
        Args:
            num_samples:

        Returns: A pytorch uint8 tensor (num_samples, num_sites)

        """
        raise NotImplementedError

    def amplitudes(self, samples, return_polar=False):
        """
        samples is (a batch) of computational basis states s. Return <s|psi>
        Args:
            samples: A pytorch uint8 tensor (..., num_sites)
            return_polar: if True, return rho + i phi, where <s|psi> = exp(rho + i phi)

        Returns: Complex of shape (...,)

        """
        raise NotImplementedError

    def conditional_logprobs(self, samples, return_phase=False):
        """
        Returns log p (s_i | s_{<i}), where p(s) = |<s|psi>|^2.
        Implemented only for autoregressive structures.

        Args:
            samples: A pytorch uint8 tensor (..., num_sites)
            return_phase: if True, also return phases of the given samples

        Returns: conditional_logprobs or (conditional_logprobs, phases), where:
            conditional_logprobs: pytorch tensor of shape (..., num_sites)
            phases: pytorch tensor of shape (...,)
        """
        raise NotImplementedError

    def full_state(self, max_batch_size=0):
        """
        Return GenericExactState, from <s|psi> for all s
        Args:

        Returns: GenericExactState

        """
        all_states = utils.all_cb_states(self.num_sites).to(
                                        next( self.parameters() ).device)
        if max_batch_size > 0:
            all_states = all_states.split(max_batch_size, dim=0)
        else:
            all_states = [all_states]
        all_amps = []
        for samples in all_states:
            all_amps.append(self.amplitudes(samples, return_polar=False))
        all_amps = utils.Complex.cat(all_amps, dim=0)
        return GenericExactState(all_amps)

    def amplitudes_in_basis(self, samples, basis):
        """
        Interpret the states as basis states of the basis `basis`.
        Return the amplitudes <s|psi>
        Args:
            samples: A pytorch uint8 tensor (..., num_sites)
            basis: An iterable of length num_sites.
                Each element is 0, 1, or 2 meaning x, y, or z

        Returns: Complex of shape (...,)

        """
        contributing, overlaps = utils.basis_change(samples, basis)
        amplitudes_compbasis = self.amplitudes(contributing)
        overlaps = overlaps.to(amplitudes_compbasis.real.device).to(
                                                amplitudes_compbasis.real.dtype)
                                                
        amplitudes = utils.Complex.einsum(  'bi,bi->b',
                                            overlaps,
                                            amplitudes_compbasis)
        return amplitudes
