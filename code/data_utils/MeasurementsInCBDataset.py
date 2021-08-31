from torch.utils.data import Dataset

import utils

class MeasurementsInCBDataset(Dataset):
    """
    Wraps around a MeasurementsDataset, and performs the basis change from the
    basis the measurement was done in to the computational basis.
    """
    def __init__(self, measurements_dataset):
        self.measurements_dataset = measurements_dataset
    def __len__(self):
        return len(self.measurements_dataset)
    def __getitem__(self, item):
        result, basis, out_dict = self.measurements_dataset[item]
        contributing_basis_states, amplitudes = utils.basis_change( result[None, :],
                                                                    basis)
        # utils.basis_change is designed to work on batches,
        # hence the padding and removing of dimensions
        return (contributing_basis_states[0], amplitudes[0], out_dict)
