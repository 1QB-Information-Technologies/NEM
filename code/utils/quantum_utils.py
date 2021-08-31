import math
import torch
import numpy as np
from utils.Complex import Complex

def long_to_bits(longs, n):
    """
    Input: torch long tensor of shape (...); n
    Output: torch uint8 tensor of shape (..., n)
    """
    idcs = torch.arange(n-1, -1, -1, dtype = torch.long, device=longs.device)
    bitstrings = ((longs[..., None] >> idcs) % 2).to(torch.uint8)
    return bitstrings

def bits_to_long(bitstrings):
    """
    Input: torch uint8 tensor of shape (..., n)
    Output: torch long tensor of shape (...)
    """
    n = bitstrings.shape[-1]
    idcs = torch.arange(n-1, -1, -1, dtype=torch.long, device=bitstrings.device)
    longs = (bitstrings.to(torch.long) << idcs).sum(dim=-1)
    return longs

def all_cb_states(num_sites):
    return long_to_bits(torch.arange(2**num_sites), num_sites)


# First axis: r = x, y, or z. Second axis: which basis state <b_r| in
# basis r, 0 meaning the positive one. Third axis:
# which computational basis state |c>. Entry: <b_r | c>
one_qubit_basis_changes = Complex(
    np.array(
        [
            [[1/math.sqrt(2), 1/math.sqrt(2)],
             [1/math.sqrt(2), -1/math.sqrt(2)]],
            [[1/math.sqrt(2), -1j/math.sqrt(2)],
             [1/math.sqrt(2), 1j/math.sqrt(2)]],
            [[1,0],
             [0,1]]
        ]
    )
).to(torch.float)

def construct_sparse_unitary(basis):
    '''
    Returns: contributing, contributions
    where contributing is of shape (2^n, num_contributing) and contains those
    computational basis els c for which <b_i |c_j> is nonzero.
    contributions is Complex and of shape (2^n, num_contributing); containing <b_i|c_j>.

    If you have the amplitudes in the comp basis as a vector, the amplitudes
    in the basis b are obtained as:
    transformed_state = (contributions * state[contributing]).sum(dim=-1)

    '''
    contributing, contributions = basis_change(all_cb_states(len(basis)), basis)
    contributing = bits_to_long(contributing)
    return contributing, contributions



def construct_full_unitary(basis):
    '''
    Constructs the unitary that changes from the given local basis to the
    computational basis. Let |b_i> be the basis elements of the given basis.
    They are numbered binary ascending, i.e. |b_0> is the product state with
    negative eigenvalue at all sites, for the sigmas specified by basis.
    Let |c_j> be the basis elements of the computational basis.
    |c_0> being all down, and |c_{2^n-1}> being all up.
    Then this method returns the matrix U_{ij} = <b_i | c_j>.

    Arguments:
        basis: a pytorch integer tensor of length n. 0 means sigma-x basis,
            1 means sigma-y, 2 means sigma-z.
    Returns:
        a 2^n x 2^n Complex pytorch tensor
    '''
    out = Complex(torch.ones((1,1)))
    for b in basis:
        out = Complex.einsum('ij,kl->ikjl', out, one_qubit_basis_changes[b])
        out = out.reshape( (2*out.shape[0], 2*out.shape[0]))
    return out

def basis_change(samples, basis):
    '''
    Each sample is understood as a basis vector in basis. For each sample,
    construct the list of computational basis vectors with which it has
    nonzero overlap, as well as the overlaps. So roughly, this returns the
    lines of the full unitary from construct_full_unitary which are specified
    by samples, in a sparse format.

    Arguments:
        samples: pytorch integer tensor (num_samples, num_sites)
        basis: pytorch integer tensor (num_sites)
    Returns: (contributing, overlaps)
        contributing: comp basis elements with nonzero overlap.
            pytorch integer tensor (num_samples, num_contributing, num_sites)
        overlaps: Complex tensor (num_samples, num_contributing)

    The returns satisfy:
        < sample | contrib_i> = overlap_i
        < sample | = \sum_i overlap_i <contrib_i|
    '''

    num_samples, num_sites = samples.shape

    contributing_basis_els = samples.reshape((num_samples, 1, num_sites))
    overlaps = Complex(torch.ones((num_samples, 1)))
    for s, b in enumerate(basis):
        if b != 2: #if the current qubit's state isn't in the z basis
            this_site_basis_change = one_qubit_basis_changes[b, contributing_basis_els[:, :, s].long(),
                                     :]
            contributing_basis_els = torch.cat(2 * [contributing_basis_els[:, :, None, :]], dim=2)
            contributing_basis_els[:, :, 0, s] = 0
            contributing_basis_els[:, :, 1, s] = 1

            overlaps = Complex.einsum('ba,bat->bat', overlaps, this_site_basis_change)
            overlaps = overlaps.reshape((num_samples, -1))
            contributing_basis_els = contributing_basis_els.reshape((num_samples, -1, num_sites))

    return contributing_basis_els, overlaps
