from physics_models import PhysicsModel
import torch
import utils

from qiskit.aqua.operators import PauliOp, SummedOp, OperatorBase
from qiskit.quantum_info import Pauli
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
from qiskit.extensions import HamiltonianGate
from typing import Optional

import math

class LatticeSchwinger(PhysicsModel):
    '''
    1D open boundary conditions lattice Schwinger model. Following 1810.03421
    (Except that the sum with prefactor g in (C4) goes over all sites, does not
    exclude the last site)
    '''
    def __init__(self,
                    num_sites: int,
                    schwinger_w:float = 1.0,
                    schwinger_m:float =-0.7,
                    schwinger_g:float =1.0,
                    schwinger_eps0:float =0.0):

        super(LatticeSchwinger, self).__init__()
        self.name = "LatticeSchwinger"
        self.num_sites = num_sites
        assert self.num_sites % 2 == 0
        self.schwinger_w = schwinger_w
        self.schwinger_m = schwinger_m
        self.schwinger_g = schwinger_g
        self.schwinger_eps0 = schwinger_eps0

        self.qiskit_h = LatticeSchwinger.get_schwinger_h_qiskit(num_sites=num_sites,
                                                                schwinger_w = schwinger_w,
                                                                schwinger_g = schwinger_g,
                                                                schwinger_eps0 = schwinger_eps0,
                                                                schwinger_m = schwinger_m)

        self.sp_hamiltonian = self.qiskit_h.to_spmatrix()

    def get_hamiltonian_lines(self, samples):
        samples_long = utils.bits_to_long(samples)
        # Separate the sparse matrix for the sampled in samples
        contributing_elements = [self.sp_hamiltonian[s_long] for s_long in samples_long]
        # Save the indicides (correspond to states) from non-zero elements of H[sample]
        contributing_states = [torch.tensor(c_elements.indices, dtype=torch.int64) for c_elements in
                               contributing_elements]
        # Save the non-zero elements of H[sample]
        contributing_data = [utils.Complex(c_elements.data).to(samples.device).to(dtype=torch.float) for c_elements
                             in contributing_elements]

        # Need to "pad" the lists so that the tensor has a defined shape
        # and each array is the same length
        # 1. Given the samples, find the the max number of contributing states (non-zero hamiltonian elements)
        max_contributing_states = max([len(c) for c in contributing_states])
        # 3. 'Pad' the shorter arrays with the 0 c.b state
        contributing_states_long_pd = torch.stack(
            [torch.cat([s, torch.zeros(max_contributing_states - len(s), dtype=torch.int64).to(samples.device)]
                       , 0) for s in contributing_states])
        # 3. 'Pad' the shorter data arrays with zeros so the 'padded' 0 c.b states
        # do not change the Hamiltonian dynamics
        contributing_data_pd = utils.Complex.stack([utils.Complex.cat(
            [data,
             utils.Complex(torch.zeros(max_contributing_states - len(data), dtype=torch.float).to(samples.device))])
            for data in contributing_data])

        # Convert the sampled states from long form (0,1,2,3...2^n) to bit strings [0010...N]
        contributing_states_pd = utils.long_to_bits(contributing_states_long_pd, self.num_sites)

        return contributing_states_pd, contributing_data_pd

    @staticmethod
    def get_resource_h_qiskit(num_sites: int,
                              j: float = 1.,
                              b: float = 10.,
                              alpha: float = 1.) -> SummedOp:
        """
        Return resource Hamiltonian, according to eq (B2) of 1810.03421v2, as a flat SummedOp.
        This resource Hamiltonian preserves the z symmetry only to first order in B/J0.
        The default value of alpha is from app. J ibid.
        The default value for B is from the comment B/J0 ~ 10 in app. C ibid.
        (In absolute terms, 1806.05737 lists J ~ 400/s, but normalize it to 1).
        :param num_sites:
        :param j: J0
        :param b: B
        :param alpha:
        :return:
        """

        terms = []
        # construct XX terms
        for i in range(num_sites-1):
            for k in range(i+1, num_sites):
                coupling = j*math.pow(abs(k-i), -alpha)
                paulistring = i*'I' + 'X' + (k-i-1) * 'I' + 'X' + (num_sites-k-1) * 'I'
                # if i==0, we want no leading Is. If k==num_sites-1, we want no trailing Is.
                # Total length should be num_sites.

                terms.append(coupling * PauliOp(Pauli.from_label(paulistring)))

        # construct B term
        for i in range(num_sites):
            paulistring = i*'I' + 'Z' + (num_sites-1-i)*'I'
            terms.append(b*PauliOp(Pauli.from_label(paulistring)))

        return SummedOp(terms)

    @staticmethod
    def get_schwinger_h_qiskit(num_sites: int,
                               schwinger_w: float = 1.0,
                               schwinger_g: float = 1.0,
                               schwinger_eps0: float = 0.0,
                               schwinger_m: float = -0.7) -> SummedOp:
        """
        Return schwinger h as a qiskit operator. Follow equation (C1) from
        self-verifying paper.

        :param num_sites:
        :param schwinger_w:
        :param schwinger_g:
        :param schwinger_eps0:
        :param schwinger_m:
        :return: Schwinger Hamiltonian as a SummedOp of AbelianSummedOps.
        """

        lambda_x_summands = []
        lambda_y_summands = []

        for j in range(num_sites - 1):
            x_pauli_string = j * 'I' + 'XX' + (num_sites - 2 - j) * 'I'
            lambda_x_summands.append(
                .5 * schwinger_w * PauliOp(Pauli.from_label(x_pauli_string)))
        lambda_x = SummedOp(lambda_x_summands, abelian=True)

        for j in range(num_sites - 1):
            y_pauli_string = j * 'I' + 'YY' + (num_sites - 2 - j) * 'I'
            lambda_y_summands.append(
                .5 * schwinger_w * PauliOp(Pauli.from_label(y_pauli_string)))
        lambda_y = SummedOp(lambda_y_summands, abelian=True)

        lambda_z_summands = []
        for j in range(num_sites):  # mass terms
            mass_pauli_string = j * 'I' + 'Z' + (num_sites - 1 - j) * 'I'
            lambda_z_summands.append(
                .5 * schwinger_m * (-1) ** (j + 1) * PauliOp(Pauli.from_label(mass_pauli_string))
            )
            # The paper's indices start at 1, ours at 0. Hence the paper has (-)^j and we need (-)^(j+1)

        for j in range(num_sites):
            # Construct the sum in the big brackets that's going to be squared
            inner_sum_terms = [schwinger_eps0 * PauliOp(Pauli.from_label(num_sites * 'I'))]
            # sum up to and including j. E.g., when j=0, we still want the sum to have one term.
            for l in range(j+1):
                z_term = PauliOp(
                    Pauli.from_label(
                        l * 'I' + 'Z' + (num_sites - 1 - l) * 'I'
                    )
                )
                inner_sum_terms.append(-.5 * z_term)
                sign_term = (-1) ** (l + 1) * PauliOp(Pauli.from_label(num_sites * 'I'))
                inner_sum_terms.append(-.5 * sign_term)

            inner_sum = SummedOp(inner_sum_terms).reduce()  # reduce collapses duplicates

            square = (schwinger_g * inner_sum @ inner_sum).reduce().reduce()
            # call reduce several times to get a flat sum with unique terms
            lambda_z_summands.append(square)

        lambda_z = SummedOp(lambda_z_summands, abelian=True).reduce().reduce()
        # Make sure lambda_z has the abelian flag set. Can't set the flag, so reconstruct the operator.
        lambda_z = SummedOp(lambda_z.oplist, abelian=True)

        h = SummedOp([lambda_x, lambda_y, lambda_z])
        return h

    @staticmethod
    def get_order_parameter_operator_qiskit(num_sites: int) -> SummedOp:
        """
        Returns the operator the expectation value of which is the order parameter,
        as given in Kokail et al 2019, at the end of section II.
        :return:
        """
        terms = []
        for i in range(num_sites):
            for j in range(i+1, num_sites):  # keep in mind our indices start at 0, the paper's at 1
                # distribute the product
                terms.append(PauliOp(Pauli.from_label(num_sites*'I')))
                terms.append((-1) ** (j + 1) * PauliOp(Pauli.from_label(j * 'I' + 'Z' + (num_sites - 1 - j) * 'I')))
                terms.append((-1) ** (i + 1) * PauliOp(Pauli.from_label(i * 'I' + 'Z' + (num_sites - 1 - i) * 'I')))
                terms.append((-1) ** (i + 1 + j + 1) * PauliOp(Pauli.from_label(
                    i * 'I' + 'Z' + (j - i - 1) * 'I' + 'Z' + (num_sites - 1 - j) * 'I'
                )))

        order_parameter_operator: SummedOp = 1/(2*num_sites*(num_sites-1)) * SummedOp(terms)
        order_parameter_operator = order_parameter_operator.collapse_summands()
        return order_parameter_operator

    @staticmethod
    def get_qiskit_noise_model(depolarizing_lambda: float = 0.02,
                               basis_gates = ['rx', 'rz', 'unitary', 'id']) -> NoiseModel:
        """
        Return a noise model: Depolarizing error all the basis gates except unitary.
        Assigning an error to unitary does not work as expected, so we'll do a
        round of 'ids' after the unitary and assign errors to those.
        :param depolarizing_lambda: Default is chosen so that the initial Neel state
        (which is prepared with an x on every other qubits) for 20 qubits has
        roughly 91% fidelity, as in app J.
        :return:
        """
        noise_model = NoiseModel(basis_gates=basis_gates)
        for basis_gate in basis_gates:
            if basis_gate not in ['unitary', 'hamiltonian']:
                noise_model.add_all_qubit_quantum_error(
                    depolarizing_error(depolarizing_lambda, num_qubits=1), basis_gate)

        return noise_model

    @staticmethod
    def get_circuit(num_sites: int,
                    initial_state: str,
                    entangling_h: Optional[OperatorBase],
                    num_layers: int,
                    identities_after_evol: bool = True,
                    barriers: bool = False,) -> QuantumCircuit:
        """
        Returns a parametric circuit built from alternating layers of single-qubit zs
        (with parameters identified to keep symmetries)
        and evolution with entangling_h.

        :param num_sites:
        :param initial_state: 'up_down' or 'down_up'. The 'up' in 'up_down' refers
            to the qubit which will be mapped to the first bit of the measurement
            result bitstring. This is the one with the largest index in the circuit.
        :param entangling_h: The H to evolve by.
        :param num_layers:
        :param identities_after_evol: if True, do an id on each qubit after the
            entangling evolution. This is a hack because applying noise to the
            unitary doesn't work as expected, so we'll apply it on the id instead.
        :param barriers: if True, do a barrier after every half-layer.
        :return:
        """
        assert initial_state in ['up_down', 'down_up']
        assert num_sites % 2 == 0
        circuit = QuantumCircuit(num_sites)

        # There is a gotcha to avoid: the qubit 0, after sampling / converting to a statevector,
        # corresponds to the rightmost element of the sample / LSB of the statevector index.
        # So, odd/even sites are exchanged in the numbering of the circuit vs. the numbering of samples/basis states.

        if initial_state == 'up_down':
            for q in range(0, num_sites, 2):
                circuit.rx(math.pi, q)
        else:
            for q in range(1, num_sites, 2):
                circuit.x(q)

        for l in range(num_layers):
            t = Parameter(f't_{l:02}')
            circuit.append(HamiltonianGate(entangling_h, t), qargs=range(num_sites))
            if identities_after_evol:
                for q in range(num_sites):
                    circuit.id(q)
            if barriers:
                circuit.barrier()
            for q in range(num_sites // 2):
                p = Parameter(f'phi_{l:02}_{q:02}')
                circuit.rz(p, q)
                circuit.rz(-p, num_sites - 1 - q)
            if barriers:
                circuit.barrier()
        return circuit
