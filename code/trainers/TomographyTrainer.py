from typing import Tuple, Optional
import torch
import utils

#Typing imports
from exact_solvers import GenericExactState
from data_utils import MeasurementsInCBDataset
from nqs_models import NeuralQuantumState
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter


class NQSTomographyTrainer():
    '''
    The neural quantum state tomography algorithm, first introduced
    here, https://arxiv.org/abs/1703.05334, trains a neural quantum state (NQS)
    ansatz to represent the approximate ground state prepared by a noisy
    quantum device.
    '''
    def __init__(self,
                 train_dl: MeasurementsInCBDataset,
                 val_dl: MeasurementsInCBDataset,
                 nqs_model: NeuralQuantumState,
                 optimizer: Optimizer,
                 target_state: Optional[GenericExactState]=None,
                 writer:SummaryWriter=None,
                 max_epochs:int=100):
        '''
        Args:
            train_dl: The training data that will be used to update the NQS Parameters
                during optimization.
            val_dl: The validation data that will be used to monitor the training optimization
            nqs_model: The NQS object that will be trained
            optimizer: The optimizer for NQST
            target_state: The target state. This object is not used in training and is optional.
                If the target state is given, the validation step will also compute the Infidelity
                to the target state.
            writer: The tensorboard writer that saves training information
            max_epochs: Maxiumum number of training epochs to be executed
        Early Breaking Criteria
            In order to avoid overfitting, we implement an early stopping check.
            If the results from the current epoch are better than the previous best,
            then we save those results best_infidelity, best_kl, best_epoch.
            If the kl-divergence does not improve for 100 epochs,
            we stop the training early.

            best_infidelity: Infidelity of the result with the lowest KL
            best_kl: Lowest KL-divergence during training
            best_epoch: Epoch of the result with the lowest KL-divergence
        '''
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.nqs = nqs_model
        self.optimizer = optimizer
        self.target_state = target_state
        self.writer=writer
        self.max_epochs = max_epochs
        self.device = next(nqs_model.parameters()).device
        #Tomography training counters and early breaking criteria
        self.best_infidelity = 1
        self.best_kl = 100
        self.best_epoch = 0

    def train_one_epoch(self) -> None:
        '''
        Training update for one epoch of neural quantum state tomography. Updates the
        neural quantum state parameters based on the kl divergence. See:
        arXiv 1703.05334

        '''
        for contributing_basis_states, amplitudes, info_dict in self.train_dl:
            contributing_basis_states = contributing_basis_states.to(self.device)
            amplitudes = amplitudes.to(self.device)

            amps = utils.Complex.einsum('ij,j->i',
                                        amplitudes,
                                        self.nqs.amplitudes(contributing_basis_states,
                                                            return_polar=False))

            logprobs = amps.norm2().log()

            loss = -logprobs.mean()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def validate(self, epoch:int) -> Tuple[float, float]:
        '''
        Validation function to see how well NQST training is performing
        ------
        Inputs:
        epoch : int | the training epoch
        ------
        '''
        num_samples = 0
        loss = 0
        kl = 0
        with torch.no_grad():
            for contributing_basis_states, amplitudes, info_dict in self.val_dl:
                contributing_basis_states = contributing_basis_states.to(self.device)
                amplitudes = amplitudes.to(self.device)
                num_samples += amplitudes.shape[0]
                amps = utils.Complex.einsum('ij,j->i',
                                            amplitudes,
                                            self.nqs.amplitudes(contributing_basis_states,
                                                                return_polar=False))
                logprobs = amps.norm2().log()
                loss += float(-logprobs.sum())
                kl += float(-logprobs.sum())
            loss /= num_samples
            kl /= num_samples

            if self.writer != None:
                self.writer.add_scalar('Val loss', loss, epoch)
                self.writer.add_scalar('Val KL', kl, epoch)

            if self.target_state != None:
                infidelity = 1-self.target_state.fidelity_to(self.nqs.full_state())
                if self.writer != None:
                    self.writer.add_scalar('Infidelity', infidelity, epoch)
                print(f"Epoch:{epoch}. Val KL:{kl}. Infidelity to Groundstate:{infidelity}.")
            else:
                print (f"Epoch:{epoch}. Val KL:{kl}. ")

            return kl, infidelity

    def train(self) -> None:
        '''
        Train the NQST where each training step is executed in
        self.train_one_epoch(epoch). Each training step minimizes
        the KL-divergence, which minimizes the "distance"
        between the probability distribution of the measurement data
        (approximated by the measurement samples) and the
        probability distribution represented by the NQS. The goal is to train
        the NQS such that it accurately represents the underlying probability
        distribution of the input measurement data.
        '''

        for epoch in range(self.max_epochs):

            self.train_one_epoch()

            val_kl, val_inf = self.validate(epoch)

            #Early stopping criteria
            if val_inf < self.best_infidelity:
                self.best_infidelity = val_inf
            if val_kl < self.best_kl:
                self.best_kl = val_kl
                self.best_epoch = epoch
            else:
                if epoch >= self.best_epoch + 100:
                    break
