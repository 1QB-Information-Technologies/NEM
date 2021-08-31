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
print('#                       Construct H2 PhysicsModel                       #')
print('#========================================================================#')
# Choose which bond length and run perform neural error mitigation on
bl = 0.6
run = 2
mol = physics_models.H2molecule(bond_length=bl,
                            mapping='bravyi_kitaev')

groundstate = mol.get_ED_groundstate()
num_sites = mol.num_qubits

print('#========================================================================#')
print('#           Load and Prepare VQE Measurements for Tomography             #')
print('#       From numerically simulated VQE with depolarizing noise           #')
print('#========================================================================#')

vqe_data_dir="/../data/H2molecule/DepolarizingNoise/"
data_file_path = sys.path[0] + vqe_data_dir + f"BL_{bl}/Run{run}/"

vqe_results             = np.load(data_file_path + "Energy_data.npz",
                                                allow_pickle=True).get('arr_0')

tomography_measurements = np.load(data_file_path + "Measurement_data.npz",
                                                allow_pickle=True).get('arr_0')

#VQE Results
energy_vqe = vqe_results.item()['energy_vqe']
fidelity_vqe = vqe_results.item()['dm_fidelity']

# Use the measurements taken at the end of VQE (See Methods Section D for additional
# information about the measurements taken for tomography)
tomography_measurements_dict = tomography_measurements.item()['tomography_measurements_dict']
bases = data_utils.str_to_list(tomography_measurements_dict.keys())
print ("Which measurements were taken", tomography_measurements_dict.keys())

print(f"VQE energy {energy_vqe} (computed) + {mol.shift} (shift)")
print (f"VQE Fidelity {fidelity_vqe}")

print ('bases = ', bases)
num_samples_per_basis = 300

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
epsilon = 0.05
vmc_iters = 1000
constant_reg_schedule = lambda iter : epsilon if iter < vmc_iters//2 else 0.0
nemtrainer = NeuralErrorMitigationTrainer(nqs_model=nqs_transformer,
                                            measurement_data=measurement_data,
                                            physics_model=mol,
                                            #NQST Parameters
                                            nqst_max_epochs=100,
                                            nqst_lr=1e-2,
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
