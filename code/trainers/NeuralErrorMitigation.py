import torch
import numpy as np

from trainers import VMCTrainer, NQSTomographyTrainer
import trainers

#Typing imports
from exact_solvers import GenericExactState
from data_utils import MeasurementsDataset
from nqs_models import NeuralQuantumState
from physics_models import PhysicsModel

from torch.utils.tensorboard import SummaryWriter
from typing import Union, Optional
import data_utils

cpu = torch.device('cpu')

class NeuralErrorMitigationTrainer:
    '''
    The neural error mitigation (NEM) method. NEM, as described in
    https://arxiv.org/abs/2105.08086 is an error mitigation method for quantum
    simulation tasks involving finding the ground state. It is composed of two
    steps:
    (A) First, we perform neural quantum state tomography (NQST) to
        train a neural quantum state (NQS) ansatz to represent the
        approximate ground state prepared by a noisy quantum device,
        using experimentally accessible measurements.
        - NeuralErrorMitigationTrainer uses NQSTomographyTrainer for this step

    (B) We then apply the variational Monte Carlo algorithm (VMC)
        on the same neural quantum state ansatz (which we call the
        NEM ansatz) to improve the representation of the unknown
        ground state.
        - NeuralErrorMitigationTrainer uses VMCTrainer for this step
    '''
    def __init__(self,
                nqs_model:NeuralQuantumState,
                physics_model:PhysicsModel,
                #Neural quantum state tomography parameters
                measurement_data:MeasurementsDataset,
                nqst_max_epochs:int=100,
                nqst_lr:float=1e-3,
                nqst_batch_size:int=128,
                #VMC Parameters
                vmc_lr:float=1e-3,
                vmc_iterations:int=1000,
                vmc_batch_size:int=256,
                vmc_epsilon:Union[float,callable]=0.0,
                vmc_regularization_type:Union['L1','L2','Entropy','None']='None',
                vmc_log_every:int=1,
                vmc_eval_every:int=100,
                #logging directories
                seed:int=0,
                logdir:Optional[str]=None
                ):
        '''
        The neural error mitigation arguements can be organized into
        general, NQST and then VMC args.

        ARGS:
        General Args:
            nqs_model: The wavefunction ansatz based off a neural network
                (neural quantum state) to be trained in NQST and VMC.
                physics_model: The physical system which we are simulating

        Neural Quantum State Tomography (NQST) Args:
            measurement_dataset: The measurement dataset used for NQST training
            nqst_num_samples_per_basis: The number of samples per measurement
                basis chosed in the measurement_dataset
            nqst_max_epochs: Maximum number of training epochs in NQST training
            nqst_lr: The learning rate for the Adam optimizer during NQST training.
            nqst_batch_size: Number of samples to include in on batch of training data

        Variational Monte Carlo (VMC) Args:
            vmc_lr: Learning rate for the VMC training optimizer
            vmc_iterations: The total number of parameter update iterations
            vmc_batch_size: The total number of samples to use to compute the
                parameter update step.
            vmc_epsilon: The regularization schedule for VMC. If vmc_epsilon is
                an integer, then the regularization will be constant. If it is
                a callable funciton (as a function of iteration), then the
                regularizer will follow the callable function.
            vmc_regularization_type:Union['L1','L2','Entropy','None']='None',
            vmc_log_every: How often the writer (SummaryWriter) should log the training
                results
            vmc_eval_every: # TO-DO fill in definition

        Extra Args:
            seed: Seed for reproducability
            logdir: Where to save the writer information

        '''

        self.nqs_model=nqs_model  # Neural Quantum State
        self.data=measurement_data # Measurement dataset used for tomography
        self.physics_model=physics_model # Phsyics model being studied (used in VMC)
        self.target_state=physics_model.get_ED_groundstate() # Target Ground state (Not required for training)
        self.seed=seed # Seed
        self.logdir=logdir # Log directory for SummaryWriter
        if self.logdir==None:
            print ("No log directory given for the tensorboard writer")
            print ("Will not log results")
        # Neural quantum state tomography training parameters
        self.nqst_lr=nqst_lr # tomography optimizer learning rate
        self.nqst_max_epochs=nqst_max_epochs # max number of trianing epochs
        self.nqst_batch_size=nqst_batch_size # nqst size of training batches
        num_bases = len(np.unique(measurement_data.bases,axis=0))
        self.nqst_num_samples_per_basis = int( len(self.data.samples)/num_bases )
        # VMC training Parameters
        self.vmc_lr = vmc_lr # vmc optimizer learning rate
        self.vmc_iterations = vmc_iterations # vmc parameter iterations
        self.vmc_batch_size = vmc_batch_size # vmc batch size
        self.vmc_epsilon = vmc_epsilon # regularization coefficient (int or callable (schedule))
        self.vmc_regularization_type = vmc_regularization_type # Type of regularizer
        self.vmc_log_every = vmc_log_every # how many iterations to log
        self.vmc_eval_every = vmc_eval_every
        # Create instances NQST trainer and VMC trainer
        #   These trainers are intialized in train() function
        self.nqst_trainer = None # Holder for the tomogrpahy trainer
        self.vmc_traimer = None # Holder for the VMC trainer

        # Final neural quantum states - assined in train()
        self.final_nqst_state:GenericExactState = None # Neural quantum state tomography result
        self.final_errmit_state:GenericExactState = None # Final Neural error mitigated result

    def _initialize_tomography(self) -> None:
        '''
        Initialize the NQSTomographyTrainer using the input arguements from
        __init__. Here, the self.measurement_dataset is transformed into the
        computational basis states and split into training data and validation data.
        Additionally, if the self.logdir is not None, then we construct the
        NQST writer which will log the optimizaiton results.
        '''
        #=============================#
        # Tomography Measurement Data #
        #=============================#
        ds = data_utils.MeasurementsInCBDataset(self.data)

        train_ds, val_ds = torch.utils.data.random_split( ds,
                            [int(.8*len(ds)), len(ds)-int(.8*len(ds))] )

        train_dl = data_utils.MeasurementsInCBDataLoader(train_ds,
                                                        num_workers=4,
                                                        shuffle=True,
                                                        batch_size = self.nqst_batch_size)

        val_dl = data_utils.MeasurementsInCBDataLoader(   val_ds,
                                                        num_workers=4,
                                                        shuffle=True,
                                                        batch_size = self.nqst_batch_size)

        optimizer = torch.optim.Adam(self.nqs_model.parameters(), lr=self.nqst_lr)

        if self.logdir == None:
            writer = None
        else:

            subdir = f'{self.physics_model.name}_{self.physics_model.bond_length}' \
                    + f'_{self.physics_model.num_qubits}_{self.nqst_num_samples_per_basis}' \
                    + f'_{self.seed}_{self.nqst_lr}_Tomography/'

            writer = self.construct_writer(subdir=subdir)

        nqst_trainer = NQSTomographyTrainer(train_dl = train_dl,
                            val_dl = val_dl,
                            nqs_model = self.nqs_model,
                            optimizer = optimizer,
                            target_state=self.target_state,
                            writer=writer,
                            max_epochs=self.nqst_max_epochs)

        self.nqst_trainer = nqst_trainer

    def _initialize_vmc(self) -> None:
        '''
        Initialize the VMCTrainer with regularization using the input arguements from
        __init__. This must be done after the tomography step is completed so that
        VMC starts after the NQS parameters have been first trained using NQST. Here, we also
        define the logging writer.
        '''
        if self.logdir == None:
            writer = None
        else:
            logsubdir   = f'{self.physics_model.name}_' \
                            f'{self.physics_model.bond_length}_{self.seed}_' \
                            f'{self.vmc_batch_size}_VMC/'

            writer      = self.construct_writer(subdir = logsubdir)
        # Callbacks
        logging_callbacks = [trainers.VMC_logging_callbacks.EnergyLoggingCallback(writer)]
        eval_callbacks = []


        vmc_trainer = VMCTrainer(
            physics=self.physics_model,
            model=self.nqs_model,
            logging_callbacks = logging_callbacks,
            eval_callbacks = eval_callbacks,
            epsilon = self.vmc_epsilon,
            regularizer = self.vmc_regularization_type,
            batch_size = self.vmc_batch_size,
            learning_rate = self.vmc_lr,
            num_iters = self.vmc_iterations,
            log_every = self.vmc_log_every,
            eval_every = self.vmc_eval_every
        ) #uses adam optimizer

        self.vmc_trainer = vmc_trainer

    def construct_writer(self,subdir)  -> SummaryWriter:
        '''
        Construct the SummaryWriter with the logging directory information as
        well as the seed.
        '''
        writer = SummaryWriter(self.logdir + subdir)
        print(f'logdir: {self.logdir + subdir}')

        writer.add_scalar('seed', self.seed)

        return writer

    def train(self)  -> None:
        '''
        Initiate the NEM training prodecure where (A) first, the NQS is trained
        with NQST to learn the state from the measurement data. Then (B) the learned
        state is improved (error mitigated) by post-processing the NQS with VMC to
        improve the estimated ground state and ground state observables.
        '''
        #=============================#
        # NQSTomography Optimization  #
        #=============================#
        print ("Performing nerual quantum state tomogrpahy on dataset")
        self._initialize_tomography()
        self.nqst_trainer.train()
        state = self.nqs_model.full_state().to(cpu)

        #Observables
        nqst_fidelity = self.target_state.fidelity_to(state)
        nqst_energy = self.physics_model.exact_expected_energy(state).real
        print ("NQST Results:")
        print ("    Energy = ", nqst_energy)
        print ("    Fidelity = ", nqst_fidelity)

        #Save NQST state result
        self.final_nqst_state = self.nqs_model.full_state().to(cpu)

        #=============================#
        #         VMC Training        #
        #=============================#
        self._initialize_vmc()
        self.vmc_trainer.train()

        state = self.nqs_model.full_state().to(cpu)

        #Observables
        errmit_fidelity = self.target_state.fidelity_to(state)
        errmit_energy = self.physics_model.exact_expected_energy(state).real
        print ("Final Error Mitigated Results:")
        print ("    Energy = ", errmit_energy)
        print ("    Fidelity = ", errmit_fidelity)

        #Save final error mitigated result
        self.final_errmit_state = self.nqs_model.full_state().to(cpu)
