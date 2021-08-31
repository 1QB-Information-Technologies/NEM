import sys
import os
import time
import torch
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '../code'))

from trainers import NeuralErrorMitigationTrainer
import physics_models
import nqs_models
import data_utils


print ("Is GPU cuda device available = ", torch.cuda.is_available())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print ("Module using ", device, "device" )
cpu = torch.device('cpu')

t_start = time.time()
print('#========================================================================#')
print('#                       Construct LiH PhysicsModel                       #')
print('#========================================================================#')
# Choose which bond length to perform neural error mitigation on
bl = 1.0
mol = physics_models.LiHmolecule(bond_length=bl,
                            mapping='parity')

groundstate = mol.get_ED_groundstate()
num_sites = mol.num_qubits

print('#========================================================================#')
print('#           Load and Prepare VQE Measurements for Tomography             #')
print('#                  From experimental VQE on IBMQ Rome                    #')
print('#========================================================================#')

vqe_data_dir="/../data/LiHmolecule/IBMQRome/"
data_file_path = sys.path[0] + vqe_data_dir + f"BL_{bl}/Run0/"

vqe_results             = np.load(data_file_path + "Energy_data.npz",
                                            allow_pickle=True).get('arr_0')

tomography_measurements = np.load(data_file_path + "Measurement_data.npz",
                                            allow_pickle=True).get('arr_0')

#VQE Results
energy_vqe = vqe_results.item()['energy_vqe']

tomography_measurements_dict = tomography_measurements.item()['tomography_measurements_dict']
bases = data_utils.str_to_list(tomography_measurements_dict.keys())
print ("Which measurements were taken", tomography_measurements_dict.keys())

print(f"VQE energy {energy_vqe} (computed) + {mol.shift} (shift)")


print ('bases = ', bases)
num_samples_per_basis = 500

measurement_data = data_utils.circuit_samples_to_MeasurementsDataset(
                                    tomography_measurements_dict,
                                    bases,
                                    num_samples_per_basis)


print('#========================================================================#')
print('#            Initialize and Perform Neural Error Mitigation                #')
print('#========================================================================#')
nqs_transformer = nqs_models.TransformerWF( num_sites = mol.num_qubits,
                                            num_layers=2,
                                            internal_dimension=8,
                                            num_heads=4,
                                            dropout=0.0).to(device)
epsilon = 0.07
vmc_iters = 1200
constant_reg_schedule = lambda iter : epsilon if iter < vmc_iters//2 else 0.0
nemtrainer = NeuralErrorMitigationTrainer(nqs_model=nqs_transformer,
                                            measurement_data=measurement_data,
                                            physics_model=mol,
                                            #NQST Parameters
                                            nqst_max_epochs=1000,
                                            nqst_lr=1e-2,
                                            nqst_batch_size=128,
                                            #VMC Parameters
                                            vmc_lr=1e-2,
                                            vmc_iterations=vmc_iters,
                                            vmc_batch_size=256,
                                            vmc_epsilon=constant_reg_schedule,
                                            vmc_regularization_type='L1',
                                            logdir=None
                                            )
nemtrainer.train()

print (f"NEM took {(time.time()-t_start)/60} minutes")
