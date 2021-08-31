# Neural Error Mitigation of Near-term Quantum Simulation

Here we demonstrate the numerical implementation of neural error mitigation, a novel method that uses neural networks to
improve estimates of ground states and ground-state observables obtained using VQE on near-term quantum computers.
This method, introduced in "Neural Error Mitigation of Near-term Quantum Simulations" https://arxiv.org/abs/2105.08086, is composed of two main steps:

**(A)** First, we perform neural quantum state tomography (NQST) to
        train a neural quantum state (NQS) ansatz to represent the
        approximate ground state prepared by a noisy quantum device,
        using experimentally accessible measurements.
        - NeuralErrorMitigationTrainer uses NQSTomographyTrainer for this step

**(B)**
 We then apply the variational Monte Carlo algorithm (VMC)
        on the same neural quantum state ansatz (which we call the
        NEM ansatz) to improve the representation of the unknown
        ground state.
        - NeuralErrorMitigationTrainer uses VMCTrainer_Regularized
         for this step

![ErrorMitigation](https://user-images.githubusercontent.com/60714304/129101121-e14bf611-5b72-4b06-9c62-32cc1f458c08.png)


## Code

`physics_models`: Hamiltonians and additional structure for the physical systems studied.

`nqs_models`: Neural Quantum States

`trainers`: Training algorithms for neural error mitigation including a neural error mitigation
trainer composed of a neural quantum state tomography (NQST) trainer and a variational Monte 
Carlo (VMC) trainer.

`utils`: Utilities including a class `Complex` which represents complex tensors and operations on them by
by wrapping around PyTorch tensors. Utilities to construct unitaries implementing basis changes,
and for setting up the logging.

`exact_solvers`: Exact diagonalization solver that finds the exact ground state from the Hamiltonians 
found in `physics_models`. The class `ExactDiagonalizer`, which diagonalizes a Hamiltonian, and 
`GenericExactState`, which keeps a (non-trainable) table of amplitudes, and some methods to sample
measurement results and query the amplitudes in different bases.

`data_utils`: Measurement data utilities for neural quantum state tomography.

## Data
Contains the variational quantum simulator measurement data for the systems studied in the paper.

`H2molecule`: Measurement data for the H2 molecule simulated numerically using a hardware-efficient 
variational quantum eigensolver ansatz with depolarizing noise.

`LiHmolecule`: Measurement data for the LiH molecule simulated numerically using a hardware-efficient 
variational quantum eigensolver ansatz with depolarizing noise as well as measurement data 
obtained from VQE on IBMQ's Rome device.

`LatticeSchwinger`: Measurement data for the lattice Schwinger model simulated numerically with depolarizing noise.

## Tutorials
Tutorials to demonstrate the application of neural error mitigation to the variational simulation of physical systems studied in "Neural Error Mitigation of Near-term Quantum Simulations." More specifically, improving estimates of the ground state and ground state observables for the electronic structure of LiH and H2 molecules as well as the lattice Schwinger model. Additionally, demo files are included to show how the variational simulations are implemented.

In order to reproduce the results of our paper, one would perform neural error mitigation (with hyperparameter values listed in Table S1) from the VQE data saved in `data` for each parameter value and run.

### Neural Error Mitigation
`NeuralErrorMitigation_LiHmolecule_ibmqrome`: We demonstrate the application of NEM to the estimation of the LiH molecular ground states prepared experimentally using VQE on IBM’s five-qubit chip, IBMQRome. Here, we input measurements taken on the final optimized VQE result (from saved data in `data/LiHmolecule/IBMQRome/` folder) in our neural error mitigation protocol. (Standard runtime ~ 4 minutes)

`NeuralErrorMitigation_LiHmolecule_depolarizing_noise`: We demonstrate the application of NEM to the estimation of the LiH molecular ground states prepared using classically simulated VQE with a depolarizing noise channel. Here, we input measurements taken on the final optimized VQE result (from saved data in `data/LiHmolecule/DepolarizingNoise/` folder) and use those measurements in our neural error mitigation protocol. (Standard runtime ~ 4 minutes)

`NeuralErrorMitigation_H2molecule`: We demonstrate the application of NEM to the estimation of the H2 molecular ground states prepared using classically simulated VQE with a depolarizing noise channel. Here, we input measurements taken on the final optimized VQE result (from saved data in `data/H2molecule/DepolarizingNoise/` folder) in our neural error mitigation protocol. (Standard runtime ~ 0.5 minute)

`NeuralErrorMitigation_LatticeSchwinger`: We demonstrate the application of NEM to the approximate ground state
of the lattice Schwinger model obtained by numerically simulating a VQE algorithm for N = 8 sites. (Standard runtime ~ 3 minutes)

### Quantum Simulations
`VariationalQuantumSimulation_LatticeSchwinger`: Demonstrate how we performed the classically simulated variational quantum simulation with a depolarizing noise chanenl for the lattice Schwinger model with N=8 sites. (Standard runtime ~ 20 seconds)

`VariationalQuantumEigensolver_QuantumChemistry` Demonstrates the performance of the hardware efficient variational quantum eigensolver classically simulated with a depolarizing noise channel to find ground state of the H2 and LiH molecule. (Standard runtime for H2 ~ 20 seconds, for LiH ~ 4 minutes)

# Environment

`environment.yml` is a list of the developer's Python environment used during development. Included are all the necessary packages and versions.

To create an environment from the `environment.yml` file:
```
conda env create -f environment.yml
```
Then activate the `NEM` environment
```
conda activate NEM
```
