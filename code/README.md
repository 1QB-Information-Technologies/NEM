# Code

This directory contains code components for performing neural error mitigation, including training and evaluating neural quantum states.

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
