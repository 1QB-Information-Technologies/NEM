from torch.utils.data import Dataset
import torch


class MeasurementsDataset(Dataset):
    """
    Contains measurement results, in different bases.
    Optionally store the logprob of getting these results.
    Logprobs are returned as 0-dimensional tensors.
    """
    @staticmethod
    def exact_from_target_state(target_state, bases, num_samples_per_basis):
        this_bases = []
        measurements = []
        logprobs = []

        for basis in bases:
            samples, logprob = target_state.sample_in_basis(basis,
                                num_samples_per_basis, return_logprobs=True)
            measurements.append(samples)
            logprobs.append(logprob)
            basis = torch.tensor(basis)[None, :].expand((num_samples_per_basis, -1))
            this_bases.append(basis)

        this_bases = torch.cat(this_bases, dim=0)
        measurements = torch.cat(measurements, dim=0)
        logprobs = torch.cat(logprobs, dim=0)

        return MeasurementsDataset(measurements, this_bases, logprobs=logprobs)

    def __init__(self, samples, bases, logprobs=None):
        self.samples = samples
        self.bases = bases
        self.logprobs = logprobs
    def __len__(self):
        return self.samples.shape[0]
    def __getitem__(self, item):
        out_dict = {}
        if self.logprobs is not None:
            out_dict['logprobs'] = self.logprobs[item]

        return (self.samples[item], self.bases[item], out_dict)
