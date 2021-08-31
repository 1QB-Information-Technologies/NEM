from typing import List, Union, Optional
import torch

from nqs_models import NeuralQuantumState
from physics_models import PhysicsModel
from torch.optim import Optimizer


class VMCTrainer:
    """
    Runs variational Monte Carlo (VMC)
    """
    def __init__(self,
                 physics:PhysicsModel,
                 model:NeuralQuantumState,
                 logging_callbacks:List,
                 eval_callbacks:List,
                 batch_size:int,
                 num_iters:int,
                 log_every:int,
                 eval_every:int,
                 learning_rate:float=None,
                 epsilon:float=None,
                 regularizer:Union['L1','L2','Entropy','None']=None,
                 optimizer:Optional[Optimizer]=None):
        '''
        Variational Monte Carlo args:

        physics: The physics model we want to find the ground state of. This is where
            we get the hamiltonian of the system that is used to compute the local energy

        model: The neural quantum state whose parameters we will optimize

        logging_callbacks: A list of logging callbacks for the SummaryWriter

        eval_callbacks: The printed callbacks during training

        batch_size: The batch size is the number of samples that will be used to
            estimate the gradient and update the NQS parameters

        num_iters: The total number of iterations to train

        log_every: How often the results should be printed

        eval_every: How often the results should be evaluated

        learning_rate: The learning rate of for the Optimizer

        epsilon: The regularization coefficient or regularization schedule

        regularizer: The regularization method to implement during VMC

        optimzier: Can input your own optimzer, if none is given, the Adam Optimizer
            is used.
        '''
        self.physics = physics
        self.model = model
        self.num_sites = physics.num_sites
        self.logging_callbacks = logging_callbacks
        self.eval_callbacks = eval_callbacks
        self.batch_size = batch_size
        self.num_iters = int(num_iters)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.log_every = log_every
        self.regularizer = regularizer
        self.eval_every = eval_every

        self.optimizer = (torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
                          if optimizer is None else optimizer)

    #===========================================================================
    # VMCTrainer with Regularizaiton functions
    #===========================================================================
    def _get_local_hamiltonians(self, samples):
        contributing, contributions = self.physics.get_hamiltonian_lines(samples)

        sample_log_amplitudes = self.model.amplitudes(samples, return_polar = True) # (b,)
        contributing_log_amplitudes = self.model.amplitudes(contributing, return_polar = True)
            # (b, k) where k indexes contributing cbs s'

        amp_fraction = (contributing_log_amplitudes - sample_log_amplitudes[:, None]).exp()

        local_hamiltonians = (amp_fraction*contributions).sum(dim=1)

        return local_hamiltonians


    def _step(self, i=0):
        with torch.no_grad():
            samples = self.model.sample(self.batch_size)
            unique_samples, counts = torch.unique(samples, dim=0, return_counts=True)
            weights = counts.type(torch.double)/self.batch_size

            local_hamiltonians = self._get_local_hamiltonians(unique_samples)
            mean_energy = (weights*local_hamiltonians.real).sum(dim=0)


        log_amplitudes = self.model.amplitudes(unique_samples, return_polar=True)
        loss_log_moduli = 2 * (local_hamiltonians.real - mean_energy) * log_amplitudes.real
        loss_phases = 2 * (local_hamiltonians.imag * log_amplitudes.imag)

        logmoduli = log_amplitudes.real
        #regularizers
        l1_of_moduli = torch.exp(-logmoduli).detach() * logmoduli
        l2_of_moduli = 2 * logmoduli
        entropy = - 4 * logmoduli.detach() * logmoduli


        regularizer = { 'L1': l1_of_moduli,
                        'L2': l2_of_moduli,
                        'Entropy': entropy,
                        'None': torch.tensor([0.0]).to(samples.device) }

        loss_vmc = (weights*(loss_phases + loss_log_moduli)).sum(dim=0)

        if callable(self.epsilon):
            epsilon = self.epsilon(i)
        else:
            epsilon = self.epsilon

        loss = loss_vmc - epsilon * ( weights*regularizer[self.regularizer] ).sum(dim=0)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)
        self.optimizer.step()
        self.optimizer.zero_grad()


        logging_dict = {
            'iter': i,
            'samples': samples.cpu().numpy(),
            'local_hamiltonians_r': local_hamiltonians.real.cpu().numpy(),
            'local_hamiltonians_i': local_hamiltonians.imag.cpu().numpy(),
            'logamplitudes_r': log_amplitudes.real.detach().cpu().numpy(),
            'loss': -self.model.amplitudes(samples,return_polar=False
                                    ).norm2().log().sum()/samples.shape[0],
            'loss_log_moduli': loss_log_moduli.detach().cpu().numpy(),
            'loss_phases': loss_phases.detach().cpu().numpy()
            }

        return logging_dict

    def train(self):
        for i in range(self.num_iters):
            logging_dict = self._step(i=i)
            if (i + 1) % self.log_every == 0:
                for logging_callback in self.logging_callbacks:
                    logging_callback(logging_dict)
            if (i + 1) % self.eval_every == 0:
                with torch.no_grad():
                    for eval_callback in self.eval_callbacks:
                        eval_callback(self.model, i)
