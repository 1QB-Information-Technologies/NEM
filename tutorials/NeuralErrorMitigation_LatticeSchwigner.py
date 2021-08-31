import sys
import os
import time
from typing import Optional
import torch
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '../code'))
from utils import Complex
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
print('#            Construct Lattice Schwinger PhysicsModel                    #')
print('#========================================================================#')
# Choose which mass and run perform neural error mitigation on
run = 0
num_sites = 8
mass = 0.1
lattice_schwinger = physics_models.LatticeSchwinger(schwinger_m = mass,
                                        num_sites = num_sites )

groundstate = lattice_schwinger.get_ED_groundstate()

print('#========================================================================#')
print('#           Load and Prepare VQE Measurements for Tomography             #')
print('#       From numerically simulated VQE with depolarizing noise           #')
print('#========================================================================#')

vqe_data_dir=f"/../data/LatticeSchwinger/{num_sites}sites/DepolarizingNoise/Mass_{mass}/Run{run}/"
data_file_path = sys.path[0] + vqe_data_dir

vqe_results             = np.load(data_file_path + "Energy_data.npz",
                                            allow_pickle=True).get('arr_0')
tomography_measurements = np.load(data_file_path + "Measurement_data.npz",
                                            allow_pickle=True).get('arr_0')

#VQE Results
energy_vqe = vqe_results.item()['energy_vqe']
fidelity_vqe = vqe_results.item()['dm_fidelity']
vqe_dm = vqe_results.item()['final_dm']
tomography_measurements_dict = tomography_measurements.item()['tomography_measurements_dict']
bases = data_utils.str_to_list(tomography_measurements_dict.keys())
print ("Which measurements were taken", tomography_measurements_dict.keys())

print(f"VQE energy {energy_vqe} (computed)")
print (f"VQE Fidelity {fidelity_vqe}")

print ('bases = ', bases)
num_samples_per_basis = 512

measurement_data = data_utils.circuit_samples_to_MeasurementsDataset(
                                    tomography_measurements_dict,
                                    bases,
                                    num_samples_per_basis)


print('#========================================================================#')
print('#            Initialize and Perform Neural Error Mitigation                #')
print('#========================================================================#')
nqs_transformer = nqs_models.TransformerWF( num_sites = lattice_schwinger.num_sites,
                                            num_layers=2,
                                            internal_dimension=8,
                                            num_heads=4,
                                            dropout=0.0).to(device)
epsilon = 0.1
vmc_iters = 400
constant_reg_schedule = lambda iter : epsilon if iter < vmc_iters//2 else 0.0
nemtrainer = NeuralErrorMitigationTrainer(nqs_model=nqs_transformer,
                                            measurement_data=measurement_data,
                                            physics_model=lattice_schwinger,
                                            #NQST Parameters
                                            nqst_max_epochs=50,
                                            nqst_lr=1e-2,
                                            nqst_batch_size=512,
                                            #VMC Parameters
                                            vmc_lr=1e-2,
                                            vmc_iterations=vmc_iters,
                                            vmc_batch_size=512,
                                            vmc_epsilon=constant_reg_schedule,
                                            vmc_regularization_type='L1',
                                            logdir=None
                                            )

nemtrainer.train()

print (f"NEM took {(time.time()-t_start)/60} minutes")

#Quantum State Results
nqst_state = nemtrainer.final_nqst_state
nem_state = nemtrainer.final_errmit_state

print('#========================================================================#')
print('#                      Compute additional Observables                    #')
print('#========================================================================#')
# Entanglement entropy: VQE, NQST, ErrMit state
# Order Parameter: VQE, NQST, ErrMit state
def get_three_site_entanglement(dm: Optional[np.ndarray] = None,
                                state: Optional[Complex] = None) -> float:
    '''
    Computes the three site entanglement entropy from either a density matrix (dm) or a state
    '''
    assert dm is not None or state is not None
    assert dm is None or state is None

    if state is not None:

        state = state.detach().numpy()
        dm = state.conjugate()[:, None] * state[None, :]

    nsites = int(np.log2(dm.shape[0] + 1))
    assert 2 ** num_sites == dm.shape[0]

    dm = dm.reshape((8, 2 ** (nsites - 3), 8, 2 ** (nsites - 3)))
    partial_tr = np.einsum('abcb->ac', dm)

    entent = 1/(1-2) * np.log(np.trace(partial_tr @ partial_tr).real) # Renyi-2

    return float(entent)
#Energy
energy_nqst = lattice_schwinger.exact_expected_energy(nqst_state).real
energy_nem = lattice_schwinger.exact_expected_energy(nem_state).real
energy_exact = lattice_schwinger.exact_expected_energy(groundstate).real
#Fidelity
fidelity_nqst = groundstate.fidelity_to(nqst_state)
fidelity_nem = groundstate.fidelity_to(nem_state)
# Order Parameter
op_matrix = lattice_schwinger.get_order_parameter_operator_qiskit(num_sites).to_matrix()
order_parameter_vqe = np.trace( op_matrix@ vqe_dm).real
order_parameter_nqst = (nqst_state.state.conjugate() @ op_matrix @ nqst_state.state).real
order_parameter_nem =(nem_state.state.conjugate() @ op_matrix @ nem_state.state).real
order_parameter_exact =(groundstate.state.conjugate() @ op_matrix @ groundstate.state).real
# Entanglement Entropy
entent_vqe = get_three_site_entanglement(dm=vqe_dm)
entent_nqst = get_three_site_entanglement(state=nqst_state.state)
entent_nem = get_three_site_entanglement(state=nem_state.state)
entent_exact = get_three_site_entanglement(state=groundstate.state)

print ("==Neural Error Mitigaiton Results==")
print ("Variational Quantum Simulation:")
print ("    Energy = ", energy_vqe, "\n    Fidelity = ", fidelity_vqe)
print ("    Order Parameter = ", order_parameter_vqe, "\n    Entanglement Entropy", entent_vqe)
print ("Neural Quantum State Tomography:")
print ("    Energy = ", energy_nqst, "\n    Fidelity = ", fidelity_nqst)
print ("    Order Parameter = ", order_parameter_nqst, "\n    Entanglement Entropy", entent_nqst)
print ("Neural Error Mitigation:")
print ("    Energy = ", energy_nem, "\n    Fidelity = ", fidelity_nem)
print ("    Order Parameter = ", order_parameter_nem, "\n    Entanglement Entropy", entent_nem)
print ("Exact Results:")
print ("    Energy = ", energy_exact)
print ("    Order Parameter = ", order_parameter_exact, "\n    Entanglement Entropy", entent_exact)
