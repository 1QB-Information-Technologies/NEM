import utils
import torch

class EnergyLoggingCallback():
    """
    Logs mean, variance and variance/mean, of the local hamiltonians encountered during training
    """
    def __init__(self, writer, print_every = 100):
        self.writer = writer
        self.print_every = print_every
    def __call__(self, logging_data):
        if self.writer is not None:
            self.writer.add_scalar('Energy/Mean',
                                    logging_data['local_hamiltonians_r'].mean(),
                                    global_step = logging_data['iter'])

            self.writer.add_scalar('Energy/Variance',
                                    logging_data['local_hamiltonians_r'].var(),
                                    global_step = logging_data['iter'])

            self.writer.add_scalar('Energy/Normalized_Variance',
                logging_data['local_hamiltonians_r'].var() / (logging_data['local_hamiltonians_r'].mean()**2) ,
                global_step = logging_data['iter'])


            sampled_states = utils.quantum_utils.bits_to_long(torch.tensor(logging_data['samples']))
            self.writer.add_histogram('Train/Samples',
                                        sampled_states,
                                        global_step = logging_data['iter'])

        if (logging_data['iter'] + 1)%self.print_every == 0:
            print(f"Iter: {logging_data['iter']}. ",
                  f"Mean energy: {logging_data['local_hamiltonians_r'].mean()}. ",
                  f"Energy variance: {logging_data['local_hamiltonians_r'].var()}.")
