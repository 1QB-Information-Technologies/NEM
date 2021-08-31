from typing import List, Optional
import torch

from qiskit.aqua.operators import PauliBasisChange
from qiskit.quantum_info import Pauli

from qiskit.circuit import QuantumCircuit
from qiskit.aqua import QuantumInstance
from qiskit.result.counts import Counts
from qiskit import transpile
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.providers.ibmq.ibmqbackend import IBMQBackend, IBMQSimulator

from data_utils import MeasurementsDataset
import utils


class TomographyMeasurements():
    '''
    The TomographyMeasurements class creates the measurement circuits needed to
    perform the measurements in the specified bases chosen for NQST. It also
    reformats the bases/samples into a dictionary that is easier to convert
    into a MeasurementsDataset (as done in circuit_samples_to_MeasurementsDataset)
    '''
    def __init__(self,
                var_form: QuantumCircuit,
                quantum_instance: QuantumInstance,
                measurement_bases: List,
                parameters: Optional[dict] = None):
        '''
        Args:
            var_form: The quantum circuit, can be parameterized or not.
                If parameterized, must input the parameter values
            quantum_instance: The QuantumInstance to execute the quantum circuits
            measurement_bases: A list of measurement bases that need to be measured.
                Given by a list of integers 0=X, 1=Y, 2=Z.
            parameters: the optimal parameters of the quantum cirucit given as a
                dictionary where parameters.keys() have the same labels as
                var_form.parameters. The parameters should be sorted from low to high
        '''
        self.var_form = var_form
        self.parameters = parameters
        # If the input circuit does not need parameters,
        # raise an error if parameters are given
        if self.var_form.num_parameters is 0 and self.parameters is not None:
            raise ValueError("Input circuit does not require parameters")
        # If the input circuit requires parameters,
        # raise an error if no parameters are given
        if self.var_form.num_parameters is not 0 and self.parameters is None:
            raise ValueError("Input circuit requires parameters")

        self.num_qubits = var_form.num_qubits
        self.quantum_instance = quantum_instance
        self.measurement_bases = measurement_bases

    def measurements_to_samples(self,
                                counts: Counts)->torch.tensor:
        '''
        Convert measurements from measurement:count form
        {'0000':100, '0001':10}
        to
        [[0,0,0,0], [0,0,0,0] ... , [0,0,0,1], [0,0,01] ]
        which is read by the neural network
        '''
        samples = []
        for state, num in zip( counts, counts.values()):
            state = torch.tensor([ int(s_i) for s_i in state])
            state_long  = utils.bits_to_long(state)
            longs = state_long.repeat(num)

            samples.append( utils.long_to_bits(longs, self.num_qubits) )
        samples = torch.cat(samples, dim=0)
        return samples

    def sample_circuit_measurements(self,
                                    measurement_bases:List[str])->dict:
        '''
        Takes samples on circuit using a prescribed measurements to be used
        for tomography.

        parameters
        -------
        params: Circuit Parameters for the base circuit
        target: Hamiltonian separated into TPB Bases (SummedOp)
        NUM_SHOTS: Number of measurements per circuit
        -----
        Returns
        Dictionary of measurement outcomes (long, torch) with their measurement string
        '''
        print ("Not using circuit batching")
        #Sample the circuit for final measurements to save for tomogrpahy
        samples_dict = {}

        for measurement_basis in measurement_bases:
            #construct circuit
            if self.parameters is not None:
                # Sort the circuit parameters in ascending order
                # Parameter(p00), Parameter(p01), etc...
                params_sorted = sorted(self.var_form.parameters, key = lambda p: p.name)
                # Assign the values into a new sorted dictionary (self.parameters)
                param_dict = dict(zip(params_sorted, self.parameters.values()))
                # Assign the input parameters to the variational circuit
                qc = self.var_form.assign_parameters(param_dict)
            else:
                qc = self.var_form
            # Define the basis we want to measure in
            basis_op = Pauli.from_label(measurement_basis)
            # Obtain the change of basis (cob) circuit for the Pauli measurements
            cob_instr_op, dest_pauli_op = PauliBasisChange().get_cob_circuit(basis_op)
            # combine with base circuit
            qc_with_post_rotations = qc.compose(cob_instr_op.to_circuit())
            qc_with_post_rotations.measure_all()
            result = self.quantum_instance.execute(qc_with_post_rotations).get_counts()

            # Sample circuit
            samples = self.measurements_to_samples(result)
            # Shuffle each outcome. This shuffles only along the first axis.
            rand_ind = torch.randperm(len(samples))
            samples = samples[rand_ind]

            # Save measurement string as dict key for samples
            samples_dict[str(measurement_basis)] = samples

        return samples_dict

    def sample_circuit_measurements_IBMQ( self,
                                    measurement_bases:List[str])->dict:
        '''
        Same function as sample_circuit_measurements but compatible with
        batching circuits for an IBMQ backend (or simulator). Batching is done
        by JobManager function when given a list of circuits. This should be used
        when calling on IBMQ Simulators and real devices with IBMQ provider credentials
        '''
        print ("Using Circuit Batching")
        #Sample the circuit for final measurements to save for tomogrpahy
        samples_dict = {}
        job_manager = IBMQJobManager()
        measurement_circuits = []
        num_mb = len(measurement_bases)
        for measurement_basis in measurement_bases:
            #construct circuit
            if self.parameters is not None:
                params_sorted = sorted(self.var_form.parameters, key = lambda p: p.name)
                param_dict = dict(zip(params_sorted, self.parameters.values()))
                qc = self.var_form.assign_parameters(param_dict)
            else:
                qc = self.var_form
            #Define the basis we want to measure in
            basis_op = Pauli.from_label(measurement_basis)
            #Obtain the change of basis (cob) circuit for the Pauli measurements
            cob_instr_op, dest_pauli_op = PauliBasisChange().get_cob_circuit(basis_op)
            #combine with base circuit
            qc_with_post_rotations = qc.compose(cob_instr_op.to_circuit())
            qc_with_post_rotations.measure_all()
            measurement_circuits.append(qc_with_post_rotations)

        #Transpile circuits
        measurement_circuits = transpile(measurement_circuits,
                                        backend=self.quantum_instance.backend)
        #Send batched, transpiled circuits to IBMQ backend
        job_set_measurements = job_manager.run(measurement_circuits,
                                backend=self.quantum_instance.backend,
                                shots = self.quantum_instance.run_config.shots)
        #Obtain the results of the job
        results = job_set_measurements.results()
        for measurement_basis, i in zip( measurement_bases, range(num_mb) ):
            #Extract counts from results and transform into samples
            samples = self.measurements_to_samples(results.get_counts(i))
            #Shuffle each outcome. This shuffles only along the first axis.
            rand_ind = torch.randperm(len(samples))
            samples = samples[rand_ind]
            #Save measurement string as dict key for samples
            samples_dict[str(measurement_basis)] = samples

        return samples_dict

    def run(self)->dict:
        #Convert bases to measurement string

        measurement_bases_string = list_to_str(
                                        self.measurement_bases)
        # measurement_bases_string = list_to_str(self.measurement_bases)
        print ("Generating Measurements for bases: \n", self.measurement_bases)
        print (measurement_bases_string)
        #Take measurements on the circuit
        if type(self.quantum_instance.backend) in [IBMQBackend, IBMQSimulator]:
            #Sample circuit measurmenets with circuit batching
            dict_measurement_samples = self.sample_circuit_measurements_IBMQ(
                                                        measurement_bases_string)
        else:
            #Sample circuit measurements without circuit batching
            dict_measurement_samples = self.sample_circuit_measurements(
                                                        measurement_bases_string)

        return dict_measurement_samples


def str_to_list(strbases: List[str]) -> List[List[int]]:
    '''
    Converts measurement bases given by a string (ex, 'XX') and converts
    it into a list of integers [[0,0]]
    '''
    char_to_int_dict = {'X': 0, 'Y': 1, 'Z': 2}
    return [[char_to_int_dict[char] for char in strbasis] for strbasis in strbases]

def list_to_str(listbases: List[List[int]]) -> List[str]:
    '''
    Converts measurement bases given by a list of integers (ex, [[0,0]]) and converts
    it into a list of strings ['XX']
    '''
    int_to_char_dict = {0: 'X', 1: 'Y', 2: 'Z'}
    return [''.join([int_to_char_dict[i] for i in listbasis]) for listbasis in listbases]


def circuit_samples_to_MeasurementsDataset(samples: dict,
                                          tomography_bases: List[List[int]],
                                          num_samples_per_basis: int)-> MeasurementsDataset:
    '''
    Converts the output of TomographyMeasurements to MeasurementsDataset
    inputs
    -------
    samples: dict, output of TomographyMeasurements().run()
    bases: List[List[int]] Measurement bases you want to train tomography on
        must be a subset of the measurement bases in samples.keys()
        num_samples_per_basis: Number of samples per basis for tomography

    returns
    -------
    MeasurementsDataset(measurements, bases)
    '''
    measurement_bases = []
    measurements = []
    #convert keys in bases_str to bases_list
    tomography_bases_str = list_to_str(tomography_bases)
    #Check that tomography_bases_str is a subset of the measurements saved in samples
    if not all(x in samples.keys() for x in tomography_bases_str):
        raise ValueError("The tomography_bases specified are not a subset of \n" \
                " the measurements saved in samples data (ie, samples.keys())")

    for tomography_basis_str, tomography_basis in zip(  tomography_bases_str,
                                                        tomography_bases) :

        samples_in_basis = samples[tomography_basis_str]

        #Reduce the number of samples
        samples_in_basis = samples_in_basis[0:num_samples_per_basis]
        #Prepare samples to put into MeasurementsDataset
        measurements.append(samples_in_basis)
        # logprobs.append(logprob)
        basis = torch.tensor(tomography_basis)[None, :].expand((len(samples_in_basis), -1))
        measurement_bases.append(basis)

    measurement_bases = torch.cat(measurement_bases, dim=0)
    measurements = torch.cat(measurements, dim=0)
    return MeasurementsDataset(measurements, measurement_bases)
