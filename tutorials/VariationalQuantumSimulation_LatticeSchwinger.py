import time
from pathlib import Path
import numpy as np
from typing import NamedTuple, List

from qiskit.aqua.algorithms import VQE
from qiskit.aqua.components.optimizers import SPSA
from qiskit.providers.aer import QasmSimulator, AerProvider
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import PauliExpectation
from qiskit.aqua.operators.converters import CircuitSampler
from qiskit.providers.aer.extensions import SnapshotDensityMatrix

import os
import sys

os.environ['QISKIT_IN_PARALLEL'] = 'TRUE'
sys.path.append(os.path.join(sys.path[0], '../code'))
print ("Schwinger VQE Path ", sys.path[0])
from physics_models import LatticeSchwinger
from data_utils import TomographyMeasurements


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
        self.t_start = time.time()

    def __call__(self,
                 eval_count: int,
                 parameters: np.ndarray,
                 energy_mean: float,
                 energy_std: float) -> None:
        self.vqe_log.eval_count_list.append(eval_count)
        self.vqe_log.parameters_list.append(parameters)
        self.vqe_log.energy_mean_list.append(energy_mean)
        self.vqe_log.energy_std_list.append(energy_std)
        if eval_count % 10 == 0:
            print(f"Circuit Eval Count {eval_count} | Energy {energy_mean} "
                  f"| Time from start {(time.time() - self.t_start) } s")


class VQERunAndSaveConfig(NamedTuple):
    num_sites: int
    schwinger_m: float
    initial_state: str
    num_layers: int
    depolarizing_lambda: float
    num_shots: int
    num_iters: int
    save_path: Path
    a0: float
    A: int
    c0: float
    alpha: float
    gamma: float
    run_num: int = 0


def run_and_save_schwinger_vqe(
        config: VQERunAndSaveConfig):
    """
    Run and save schwinger vqe with the given parameters
    :param config:
    :return:
    """


    initial_state = config.initial_state
    schwinger_m = config.schwinger_m
    num_sites = config.num_sites
    num_layers = config.num_layers
    depolarizing_lambda = config.depolarizing_lambda
    num_shots = config.num_shots
    num_iters = config.num_iters
    save_path = config.save_path
    run_num = config.run_num
    alpha = config.alpha
    A = config.A
    a0 = config.a0
    c0 = config.c0
    gamma = config.gamma

    t0 = time.time()

    #================================================================#
    #              Construct the variational circuit                 #
    #================================================================#
    target_h = LatticeSchwinger.get_schwinger_h_qiskit(num_sites=num_sites,
                                                       schwinger_m=schwinger_m)
    resource_h = LatticeSchwinger.get_resource_h_qiskit(num_sites)
    circuit = LatticeSchwinger.get_circuit(num_sites=num_sites,
                                           initial_state=initial_state,
                                           entangling_h=resource_h,
                                           num_layers=num_layers,
                                           identities_after_evol=True,
                                           barriers=True)
    #================================================================#
    #                 Construct the noisy simulator                  #
    #================================================================#
    simulator = QasmSimulator(provider=AerProvider(),
                              method='density_matrix', max_parallel_threads=1)
    noisemodel = LatticeSchwinger.get_qiskit_noise_model(depolarizing_lambda=depolarizing_lambda,
                                                         basis_gates=['rx', 'rz', 'unitary', 'id', 'u'])
    instance = QuantumInstance(backend=simulator, basis_gates=noisemodel.basis_gates,
                               noise_model=noisemodel,
                               shots=num_shots,
                               optimization_level=0
                               )
    print ("#================================================================#")
    print ("#                            Perform VQE                         #")
    print ("#================================================================#")
    optimizer = SPSA(maxiter=num_iters,
                     skip_calibration=True,
                     c0=a0,
                     c1=c0,
                     c2=alpha,
                     c3=gamma,
                     c4=A,
                     last_avg=8)
    callback = VQELogger()
    vqe = VQE(operator=target_h,
              var_form=circuit,
              optimizer=optimizer,
              initial_point=np.zeros(len(circuit.parameters)),
              expectation=PauliExpectation(),
              quantum_instance=instance,
              callback=callback
              )
    vqe._circuit_sampler = CircuitSampler(
        instance,
        param_qobj=False)

    t = time.time()
    vqe_result = vqe.run()
    print(f'VQE finished, took {(time.time() - t) / 60} min')

    print ("#================================================================#")
    print ("#               Estimate additional observables                  #")
    print ("#================================================================#")
    # Extract final density matrix
    optimal_circuit = vqe.get_optimal_circuit()
    simulator = QasmSimulator(provider=AerProvider(),
                              method='density_matrix', max_parallel_threads=1)
    instance = QuantumInstance(backend=simulator, basis_gates=noisemodel.basis_gates,
                               noise_model=noisemodel,
                               shots=num_shots,
                               optimization_level=0
                               )
    snap = SnapshotDensityMatrix(label='snap', num_qubits=num_sites)
    optimal_circuit.append(snap, list(range(num_sites)))
    optimal_circuit_result = instance.execute(optimal_circuit)
    final_dm: np.ndarray = optimal_circuit_result.data()['snapshots']['density_matrix']['snap'][0]['value']

    # Get fidelity and energy

    tr_rho2 = np.trace(final_dm @ final_dm).real
    print(f'tr_rho2={tr_rho2}')
    schwinger = LatticeSchwinger(num_sites=num_sites, schwinger_m=schwinger_m)
    ham_matrix = schwinger.get_hamiltonian_matrix().numpy()
    final_noisy_energy = np.trace(final_dm @ ham_matrix)
    print(f'final energy vqe: {final_noisy_energy}')
    exact_gs = schwinger.get_ED_groundstate()
    exact_gs_e = schwinger.exact_expected_energy(exact_gs).numpy().real
    print(f'exact_gs_e {exact_gs_e}')
    exact_gs_np = exact_gs.state.numpy()
    final_noisy_fid = (exact_gs_np.conjugate() @ final_dm @ exact_gs_np).real
    print(f'final dm fid {final_noisy_fid}')

    results_dict = { 'parameters': vqe_result['optimal_parameters'],
                        'energy_vqe': float(final_noisy_energy),
                        'energy_exact': exact_gs_e,
                        'exact_groundstate': exact_gs_np,
                        'energy_log': np.array(callback.vqe_log.energy_mean_list),
                        'energy_std_log': np.array(callback.vqe_log.energy_std_list),
                        'parameter_log': callback.vqe_log.parameters_list,
                        'final_dm': final_dm,
                        'tr_rho2': tr_rho2,
                        'dm_fidelity': final_noisy_fid,
                        'schwinger_m': schwinger_m,
                        'layers' : num_layers,
                        'depolarizing_lambda': depolarizing_lambda,
                        'physics_model': "LatticeSchwinger",
                        'num_sites': num_sites,
                        'num_shots' : num_shots,
                        'num_iters' : num_iters,
                        'run': run_num,
                        'vqe_initial_state': initial_state,
                        }

    print ("#================================================================#")
    print ("#        Take nearly-diagonal measurements needed for NQST       #")
    print ("#================================================================#")

    measurement_bases = [num_sites * [2]]
    for i in range(num_sites - 1):
        measurement_bases.append(i * [2] + [0, 0] + (num_sites - i - 2) * [2])
        measurement_bases.append(i * [2] + [1, 1] + (num_sites - i - 2) * [2])

    samples_factory = TomographyMeasurements(var_form=optimal_circuit,
                                             quantum_instance=instance,
                                             measurement_bases=measurement_bases)
    measurement_samples_dict = samples_factory.run()

    measurements_results_dict = {
                        'tomography_measurements_dict': measurement_samples_dict,
                        'mass': schwinger_m,
                        'layers' : num_layers,
                        'physics_model' : "LatticeSchwinger",
                        'num_shots' : num_shots,
                        'run': run_num }

    print (f"One VQE run took {(time.time() - t0)/60}")
    save=False
    if save==True:
        print ("Saving Results")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        np.savez(save_path / Path("Energy_data"),
                    results_dict)

        np.savez(save_path / Path("Measurement_data"),
                    measurements_results_dict)

if __name__ == '__main__':
    num_sites = 8

    for initial_state, schwinger_m in zip(3*['down_up'] + 3*['up_down'],
                                        [-1.5, -1.1, -0.7, -0.7, -0.3, 0.1]):

        for run_no in range(1):
            # Save tutorial VQE data into a new VQE_data file
            data_save_path = Path( sys.path[0] + f"/VQE_data/LatticeSchwinger/{num_sites}sites"\
                                 f"/DepolarizingNoise/Mass_{schwinger_m}/Run{run_no}/" )

            config = VQERunAndSaveConfig(
                        num_sites=num_sites,
                        schwinger_m=schwinger_m,
                        initial_state=initial_state,
                        num_layers=3,
                        depolarizing_lambda=0.001,
                        a0=0.1,
                        A=10,
                        c0=0.1,
                        alpha=.602,
                        gamma=.101,
                        num_shots=512,
                        num_iters=2,
                        run_num=run_no,
                        save_path=data_save_path
                        )

            run_and_save_schwinger_vqe(config)
