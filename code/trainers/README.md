# Trainers
This directory contains the training algorithms involved in performing neural error mitigation (NEM).

`NeuralErrorMitigationTrainer` performs NEM and initializes the two training algorithms - NQSTomographyTrainer and VMCTrainer.

`NQSTomographyTrainer` performs neural quantum state tomography using the measurement data obtained at the end of VQE. It updates the parameters of the NQS by minimizing the KL Divergence.

`VMCTrainer` performs variational Monte Carlo with a regularization schedule. In order to perform NEM, a NQS that has been pre-trained using the NQSTomographyTrainer must be input into the VMCTrainer.

## Callbacks
In addition to trainers, here we also construct the callback that can be logged by a TensorBoard SummaryWriter (if a log directory is provided to the NeuralErrorMitigationTrainer).

`VMC_logging_callbacks` define the energy logging callbacks for the VMCTrainer

## Additional Information
Please refer to the following sections of https://arxiv.org/pdf/2105.08086.pdf and references therein for additional information about each training algorithm. 

**Methods B:** Neural Quantum State Tomography

**Methods C:** Variational Monte Carlo

**Supplementary Information II:** Variational Monte Carlo and Regularization
