import itertools
from pathlib import Path
from typing import List, NamedTuple
import time
import os
import sys
import numpy as np
#Qiskit functionality
from qiskit.aqua.algorithms import VQE
from qiskit.providers.aer import QasmSimulator, AerProvider
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
from qiskit.transpiler import CouplingMap
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import PauliExpectation
from qiskit.aqua.components.optimizers import SPSA
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.providers.aer.extensions import SnapshotDensityMatrix

sys.path.insert(1, os.path.join(sys.path[0], '../code'))
system_path = sys.path[0]
import physics_models
import data_utils


t_start = time.time()
class VQELog(NamedTuple):
    """
    Container for a log of a vqe run. Constructed by VQELogger.
    """
    eval_count_list: List[int] = []
    parameters_list: List[np.ndarray] = []
    energy_mean_list: List[float] = []
    energy_std_list: List[float] = []

class VQELogger:
    """
    Pass an instance of this as a callback to VQE.__init__,
    and access the log in self.vqe_log when the run is done.
    """

    def __init__(self):
        self.vqe_log = VQELog()

    def __call__(self, eval_count: int,
                 parameters: np.ndarray,
                 energy_mean: float,
                 energy_std: float) -> None:
        self.vqe_log.eval_count_list.append(eval_count)
        self.vqe_log.parameters_list.append(parameters)
        self.vqe_log.energy_mean_list.append(energy_mean)
        self.vqe_log.energy_std_list.append(energy_std)
        if eval_count%50 == 0:
            print (f"Circuit Eval Count {eval_count} | Energy {energy_mean}"\
                    f" | Time from start = {( time.time()- t_start )/(60) } Minutes ")


def get_circuit(num_qubits: int,
                num_layers: int,
                edges: List[List[int]],
                barriers: bool = False) -> QuantumCircuit:
    """
    Construct a hardware efficient circuit.
    :param num_qubits:
    :param num_layers:
    :param edges: List of pairs of ints. The entangling layer is going to
                    have a CNOT for every pair.
    :param barriers: Whether to put barriers in between the sublayers.
    :return:
    """
    qc = QuantumCircuit(num_qubits)
    params = []
    # initial Euler Rotation Layer
    for i in range(num_qubits):
        for _ in range(2):  # two new parameters
            params.append(Parameter(f'p{len(params):02}'))
        # rotation with the two new parameters. Don't need the first
        # z rotation
        qc.u(params[-2], params[-1], 0, i)
    if barriers:
        qc.barrier()
    for l in range(num_layers):
        # entangling layer
        for pair in edges:
            qc.cnot(pair[0], pair[1])
        if barriers:
            qc.barrier()
        for i in range(num_qubits):
            for _ in range(3):
                params.append(Parameter(f'p{len(params):02}'))
            qc.u(params[-3], params[-2], params[-1], i)
        if barriers:
            qc.barrier()
    return qc


def perform_vqe(config):
    '''
    Function to perform VQE with depolarizing noise for the specified model
    '''
    t0 = time.time()
    # Molecule we want to simulate
    mol = config.physics_model

    groundstate = mol.get_ED_groundstate()
    groundstate_energy = mol.exact_expected_energy(groundstate).real
    print(f"Number of Tensor Product Bases: {len(mol.qubit_hamiltonian)}")
    print ("Groundstate enegy ", groundstate_energy)


    #================================================================#
    #                 Construct the noisy simulator                  #
    #================================================================#
    print ("Quantum Simulator = ", config.simulator_method)
    simulator = QasmSimulator(provider=AerProvider(),
                              method=config.simulator_method, max_parallel_threads=1)

    # Construct the noise model
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(.001, 1), 'u3')
    noise_model.add_all_qubit_quantum_error(depolarizing_error(.01, 2), 'cx')
    basis_gates = noise_model.basis_gates
    # linear circuit connectivity
    connectivity = []
    for i in range(mol.num_qubits - 1):
        connectivity.append([i, i + 1])

    qinstance = QuantumInstance(backend=simulator,
                                shots=config.num_shots,
                                coupling_map=CouplingMap(connectivity),
                                basis_gates=basis_gates,
                                noise_model=noise_model)

    #================================================================#
    #              Construct the variational circuit                 #
    #================================================================#


    circuit = get_circuit(num_qubits=mol.num_qubits,
                          num_layers=config.layers,
                          edges=connectivity,
                          barriers=False)

    print(f"num_parameters={len(circuit.parameters)} | num_layers = {config.layers}")

    print ("#================================================================#")
    print ("#                            Perform VQE                         #")
    print ("#================================================================#")
    optimizer = SPSA(maxiter=config.num_iters)

    logger = VQELogger()
    vqe = VQE(operator=mol.qubit_hamiltonian,
              optimizer=optimizer,
              var_form=circuit,
              quantum_instance=qinstance,
              callback=logger,
              expectation=PauliExpectation())

    result = vqe.run()

    optimal_parameters = result['optimal_parameters']
    energy_vqe = result['eigenvalue'].real

    print ("#================================================================#")
    print ("#               Estimate additional observables                  #")
    print ("#================================================================#")
    # Extract Fidelity from the final density matrix
    optimal_circuit = vqe.get_optimal_circuit()
    simulator = QasmSimulator(provider=AerProvider(),
                              method='density_matrix', max_parallel_threads=1)
    instance = QuantumInstance(backend=simulator, basis_gates=basis_gates,
                               noise_model=noise_model,
                               shots=config.num_shots,
                               optimization_level=0)

    snap = SnapshotDensityMatrix(label='snap', num_qubits=config.num_sites)
    optimal_circuit.append(snap, list(range(config.num_sites)))
    optimal_circuit_result = instance.execute(optimal_circuit)

    final_dm: np.ndarray = optimal_circuit_result.data()['snapshots']['density_matrix']['snap'][0]['value']

    # Get fidelity and energy
    tr_rho2 = np.trace(final_dm @ final_dm).real
    ham_matrix = mol.get_hamiltonian_matrix().numpy()
    final_noisy_energy = np.trace(final_dm @ ham_matrix)

    exact_gs_np = groundstate.state.numpy()
    final_noisy_fid = (exact_gs_np.conjugate() @ final_dm @ exact_gs_np).real

    print(f"Final Energy: {energy_vqe} Noisy Fidelity: {final_noisy_fid}")
    print(f"Took {logger.vqe_log.eval_count_list[-1]} energy evals")

    results_dict = {    'parameters': optimal_parameters,
                        'energy_vqe': energy_vqe,
                        'energy_exact': groundstate_energy,
                        'exact_groundstate': groundstate,
                        'energy_log': np.array(logger.vqe_log.energy_mean_list),
                        'energy_std_log': np.array(logger.vqe_log.energy_std_list),
                        'parameter_log': np.array(logger.vqe_log.parameters_list),
                        'final_dm': final_dm,
                        'tr_rho2': tr_rho2,
                        'dm_fidelity': final_noisy_fid,
                        'connectivity': connectivity,
                        'bond_length': config.bond_length,
                        'mapping' :config.mapping,
                        'layers' : config.layers,
                        'simulator_method' : config.simulator_method,
                        'physics_model' : config.physics_model,
                        'num_shots' : config.num_shots,
                        'num_iters' : config.num_iters,
                        'take' : config.take,
                        'run': config.run_num }

    print ("#================================================================#")
    print ("#        Take nearly-diagonal measurements needed for NQST       #")
    print ("#================================================================#")

    Weight1_combos = config.num_sites
    Weight2_combos = int(config.num_sites*(config.num_sites-1)/2)
    bases_z   = [config.num_sites * [2] for _ in range(1)]
    bases_zx  = [config.num_sites * [2] for _ in range(Weight1_combos)]
    bases_zxx = [config.num_sites * [2] for _ in range(Weight2_combos)]
    for i in range(Weight1_combos): # One X measurement
        bases_zx[i][i] = 0
    row = 0
    for n in range(config.num_sites): # Two X Measurements
        for m in range(n+1,config.num_sites):
            bases_zxx[row][n] = 0
            bases_zxx[row][m] = 0
            row += 1
    bases = list(itertools.chain(bases_z, bases_zx, bases_zxx))
    print ('Nearly-diagonal measurement bases = ', bases)

    measurement_samples_dict = data_utils.TomographyMeasurements(
                                        var_form = optimal_circuit,
                                        quantum_instance = qinstance,
                                        measurement_bases = bases).run()
    measurements_results_dict = {
                        'tomography_measurements_dict': measurement_samples_dict,
                        'measurement_bases': bases,
                        'connectivity': connectivity,
                        'bond_length': config.bond_length,
                        'mapping' :config.mapping,
                        'layers' : layers,
                        'simulator_method' : config.simulator_method,
                        'physics_model' : config.physics_model,
                        'num_shots' : config.num_shots,
                        'take' : config.take,
                        'run': config.run_num }

    print (f"One VQE run took {(time.time()-t0)/60} minutes")

    save=False
    if save==True:
        print ("Saving Results")
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path)

        np.savez(config.save_path / Path("Energy_data"),
                    results_dict)

        np.savez(config.save_path / Path("Measurement_data"),
                    measurements_results_dict)

class RunAndSaveConfig(NamedTuple):
    num_sites: int
    physics_model: physics_models.PhysicsModel
    bond_length: float
    mapping: str
    layers: int
    simulator_method: str
    noise: bool
    num_shots: int
    num_iters: int
    save_path: Path
    take: int
    run_num: int = 0


if __name__ == '__main__':


    molecule = "H2molecule"
    mapping = "bravyi_kitaev"
    simulator_method = "density_matrix"
    bond_lengths = np.array([.2])

    # Can also use this tutorial to run VQE for LiHmolecule
    # Uncomment the following code:

    # molecule = "LiHmolecule"
    # mapping = "parity"
    # simulator_method = "density_matrix"
    # bond_lengths = np.array([.4])


    total_runs = np.arange(0,1)
    layers = 1
    take = 1
    noise = True
    for bond_length in bond_lengths:
        if molecule is "H2molecule":
            physics_model = physics_models.H2molecule(
                                bond_length = bond_length,
                                mapping = mapping)
        elif molecule is "LiHmolecule":
            physics_model = physics_models.LiHmolecule(
                                bond_length = bond_length,
                                mapping = mapping)

        for run in total_runs:
            # Save tutorial VQE data into a new VQE_data file
            data_save_path = system_path + f"/VQE_data"\
                                 f"/{physics_model.name}/DepolarizingNoise/Layers{layers}"\
                                 f"/Take{take}/BL_{bond_length}/Run{run}/"
            print ("Data_save_path = ", data_save_path )
            vqe_config = RunAndSaveConfig(
                num_sites = physics_model.num_qubits,
                physics_model = physics_model,
                bond_length = bond_length,
                mapping = mapping,
                layers = layers,
                simulator_method = simulator_method,
                noise = noise,
                num_shots = 1024,
                num_iters = 250,
                save_path = Path(data_save_path),
                take = take,
                run_num = run)

            perform_vqe(vqe_config)
