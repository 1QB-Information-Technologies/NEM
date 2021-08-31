import torch
from torch.utils.data import DataLoader

import utils

def basis_change_collate_fn(list_of_items):
    contributing_basis_states_list, amplitudes_list, info_dicts = \
        [[s[i] for s in list_of_items] for i in range(3)]

    contributing_basis_states = torch.cat(contributing_basis_states_list, dim=0)

    amplitudes = utils.Complex(torch.zeros((len(list_of_items),
                                    contributing_basis_states.shape[0])))

    start_idx = 0
    end_idx = 0
    for i, amp in enumerate(amplitudes_list):
        end_idx += amp.shape[0]
        amplitudes.real[i,start_idx:end_idx]=amp.real
        amplitudes.imag[i,start_idx:end_idx]=amp.imag
        start_idx += amp.shape[0]

    info_dict = {}
    if 'logprobs' in info_dicts[0].keys():
        info_dict['logprobs'] = torch.stack([i['logprobs'] for i in info_dicts])

    return contributing_basis_states, amplitudes, info_dict


class MeasurementsInCBDataLoader(DataLoader):
    """
    DataLoader for a MeasurementsInCBDataset. An item of a MeasurementsInCBDataset
    is (contributing_basis_states, amplitudes, info_dict), and contributing_basis_states
    as well as amplitudes vary in length, depending on how many computational basis states
    the original result overlaps with. So concatenating several of these items into
    a minibatch needs to be done carefully.

    A minibatch from this dataloader is (contributing_basis_states, amplitudes, info_dict), where:
    - contributing_basis_states is obtained by concatenating contributing_basis_states
        from all the items in the minibatch. It is of shape
        (num_contributing_basis_states_total, num_sites)

    - amplitudes is of shape (minibatch_size, num_contributing_basis_states_total).
        Its first line contains amplitudes from the first item in the minibatch, etc.

    - info_dict['logprobs'], if present, is of shape (minibatch_size,) and is the
        concatenation of logprobs from the items.

    Suppose the first element of the minibatch is the measurement result <s|. Then the
    overlap <s|psi> is obtained as:
    <s|psi> = (amplitudes[0, :] * psi.amplitudes(contributing_basis_states)).sum()
    """
    def __init__(self, *args, **kwargs):
        super(MeasurementsInCBDataLoader, self).__init__(*args,
                                            collate_fn = basis_change_collate_fn,
                                            **kwargs)
