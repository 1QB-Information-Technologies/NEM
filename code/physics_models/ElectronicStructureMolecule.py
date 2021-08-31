from typing import Optional, List
import torch

from qiskit.chemistry.transformations import FermionicTransformation, FermionicTransformationType,FermionicQubitMappingType
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry import FermionicOperator
from qiskit.aqua.operators.converters import AbelianGrouper
from qiskit.aqua.operators import Z2Symmetries
#Imported for typing purposes
from qiskit.aqua.operators import SummedOp


import numpy as np
import utils
from physics_models.PhysicsModel import PhysicsModel
#Define aqua seed

class ElectronicStructureMolecule(PhysicsModel):
    '''
    ElectronicStructureMolecule is a class of Hamiltonians that
    describe electronic structure models. It constructs the QubitHamiltonian
    from PySCFDriver
    *This class is also compatible with the PhysicsModel class
    which is used during NN training and VMC by
    > Defining the get_hamiltonian_lines(samples) function by
        Converting the qubit Hamitlonian into a sparse matrix which
        is used in get_hamiltonian_lines & PhysicsModel functions
    '''
    def __init__(self,
                name: str,
                atom: str,
                mapping: ['parity', 'jordan_wigner', 'bravyi_kitaev'],
                ham_type: ["New", "Old"],
                orbital_reduction: Optional[List[int]] = None,
                freeze_core: Optional[bool] = False,
                two_qubit_reduction: Optional[bool] = False):
        '''
        Parameters
            -----------
            atom: str
                String describing the Molecular geometry of the electronic Structure
                problem. Typically composed of 'Atom1 x_coordinate y_coordinate z_coordinate; ... '
            mapping: str
                String describing the type of fermionic to qubit mapping
            ham_type: ["New", "Old"]
                one of: ['Old', 'New']
                Tells the ElectronicStructureMolecule to construct the qubit Hamiltonian
                for the molecule using the 'new' method (qubit_ops) or the 'old' one (old_qubit_ops).
                Here, "Old" is used for the H2 Hamiltonian and "New" is used for all other models.
            orbital_reduction: Optional[List[int]]
                List of orbitals to remove from the simulation. These are typically high energy orbitals
                which will not be occupied by the particles (electrons) in the simulation
            freeze_core: bool
                Boolean statement on whether or not to freeze the core orbitals
                whose occupation will not change. If True we freeze the core, if False we do not.
                If True, we will also obtain a non-zero energy contribution from the
                frozen orbitals
            two_qubit_reduction: bool
                Boolean statement on whether or not to apply the Z2Symmetries.two_qubit_reduction
                to the system. For the 'new' method this is only implemented for a 'parity' mapping.
                For the 'Old' method, it is done for all (whether or not it is physically motivated)
            Note on methods
            ----------
            In addition to computing the qubit Hamiltonian, we also group the qubit
            Hamiltonian into Tensor Product Bases as defined in arxiv.1704.05018
        '''
        super().__init__() #Initialize PhysicsModel
        self.name = name
        self.mapping = mapping
        #String that defines the atom as input into PySCFDriver
        self.atom = atom
        #Qubit Transformation Information for FermionicTransformation
        self.oribital_reduction = orbital_reduction
        self.freeze_core = freeze_core
        self.two_qubit_reduction = two_qubit_reduction
        self.ham_type = ham_type #Qubit hamitlonian contruction type
        self.shift = None #Energy shift
        self.hf_energy = None #HF energy
        self.num_qubits = None #Number of qubits to describe Hamiltonian
        self.num_particles = None #Number of particles involved in dynamics
        self.num_orbitals = None #Number of spin orbitals involved in dynamics
        #Generate the Qubit Hamiltonian given the above specifications
        grouper = AbelianGrouper()
        qubit_ops = self.old_qubit_ops() if self.ham_type == "Old" else self.qubit_ops()
        qubit_hamiltonian = grouper.convert(qubit_ops)
        #Qubit Hamiltonian is input into
        #Qiskit VQE functions
        self.qubit_hamiltonian = qubit_hamiltonian
        #Save Sparse Hamiltonian instead of 2^n * 2^n
        #matrix --> used in get_hamiltonian_lines

        self.sp_hamiltonian_csr = qubit_hamiltonian.to_spmatrix()
        sp_ham_coo = self.sp_hamiltonian_csr.tocoo()
        #pytorch sparse matrix form to be compatible with .to(device_gpu)
        self.sp_hamiltonian = torch.sparse.LongTensor(torch.LongTensor([
                    sp_ham_coo.row.tolist(), sp_ham_coo.col.tolist()]),
                    torch.Tensor(sp_ham_coo.data.real.astype(np.float32)))
        #Define the number of sites for the PhysicsModel
        self.num_sites = self.num_qubits

    def get_hamiltonian_lines(self, samples, **kwargs):
        '''
        Takes a batch of samples and outputs the contributing lines of the
        Hamiltonian and the indicies (states). We do this using the sparse matrix
        obtained from the QubitHamiltonian
        Parameters
        ----------------
        samples: Tensor output of utils.long_to_bits(samples)

        Returns
        ----------------
        Contributing_states_pd Torch.Tensor
            Lists the computational basis states with non-zero entries in Hamiltonian
            for each sampled state in samples.
        Contributing_data_pd: Complex tensor
            Lists the non-zerp Hamiltonian row elements for each sampled state in samples.
        '''
        samples_long = utils.bits_to_long(samples)

        self.sp_hamiltonian = self.sp_hamiltonian.to(samples.device)
        #Separate the sparse matrix for the sampled in samples
        contributing_elements = [ self.sp_hamiltonian[s_long] for s_long in samples_long]
        # Save the indicides (correspond to states) from non-zero elements of H[sample]
        contributing_states = [ c_elements._indices()[0].to( dtype=torch.int64 )
                                        for c_elements in contributing_elements]
        # print ("contributing states = ", contributing_states)
        # Save the non-zero elements of H[sample]
        contributing_data = [ utils.Complex( c_elements._values() ).to(samples.device).to(dtype = torch.float)
                                            for c_elements in contributing_elements]

        # Need to "pad" the lists so that the tensor has a defined shape
        # and each array is the same length
        # 1. Given the samples, find the the max number of contributing states (non-zero Hamiltonian elements)
        max_contributing_states = max( [ len(c) for c in contributing_states ] )
        # 3. 'Pad' the shorter arrays with the 0 c.b state
        contributing_states_long_pd = torch.stack(
            [ torch.cat( [ s , torch.zeros(max_contributing_states - len(s), dtype=torch.int64).to(samples.device)]
            , 0) for s in contributing_states] )
        # 3. 'Pad' the shorter data arrays with zeros so the 'padded' 0 c.b states
        # do not change the Hamiltonian dynamics
        contributing_data_pd = utils.Complex.stack([utils.Complex.cat(
            [ data , utils.Complex( torch.zeros( max_contributing_states - len(data),
                            dtype=torch.float).to(samples.device)) ])
                            for data in contributing_data])

        #Convert the sampled states from long form (0,1,2,3...2^n) to bit strings [0010...N]
        contributing_states_pd = utils.long_to_bits( contributing_states_long_pd, self.num_sites )

        return contributing_states_pd, contributing_data_pd

    def qubit_ops(self) -> SummedOp:
        '''
        New method for constructing the Qubit Hamitlonian. This method has fewer steps
        and requires fewer inputs by the user. FermionicTransformation defines the
        transformation from fermionic operators to qubit ops.
        NOTE: Only performs two_qubit_reduction if FerminonicQubitMappingType is "parity".
        **We know we can apply the two qubit reduction to H2 with "bravyi-kitaev" mapping
        **Shown in: arxiv.1704.05018
        Therefore, for H2 with "bravyi_kitaev" mapping use the OldQubitOp() method
        '''

        #Constructs molecule PySCF
        driver = PySCFDriver(atom=self.atom, unit=UnitsType.ANGSTROM,
                             charge=0, spin=0, basis='sto3g')# , max_memory=1500)
        #Define a transformation from fermions to Qubits
        transformation = FermionicTransformation(
                            transformation = FermionicTransformationType.FULL,
                            qubit_mapping=FermionicQubitMappingType(self.mapping),
                            two_qubit_reduction=self.two_qubit_reduction,
                            freeze_core=self.freeze_core,
                            orbital_reduction=self.oribital_reduction)

        #Apply Transformation from fermionic operators to qubit operators
        qubitOp, aux = transformation.transform(driver)
        #save the number of qubits needed
        self.num_qubits = qubitOp.num_qubits
        #auxillary terms which do not define the system
        self.auxillary_operators = aux
        #Save the energy shift (Nuclear + frozen + removed )
        self.shift = transformation._nuclear_repulsion_energy + transformation._energy_shift
        #Save the hartree fock energy
        self.hf_energy = transformation._hf_energy
        # self.num_particles = np.sum()
        self.num_particles = transformation.molecule_info['num_particles']
        self.num_orbitals = transformation.molecule_info['num_orbitals']

        return qubitOp

    def old_qubit_ops(self) -> SummedOp:
        '''
        Old method of constructing the qubit Hamiltonian using the
        PySCFDriver. this method requires inputting the frozen Orbitals
        and High Energy Oribtals that you want to remove by hand. In addition,
        it does all of the reduction and conversion more explicitly
        '''

        driver = PySCFDriver(atom=self.atom, unit=UnitsType.ANGSTROM,
                             charge=0, spin=0, basis='sto3g')
        molecule = driver.run()
        repulsion_energy = molecule.nuclear_repulsion_energy
        num_particles = molecule.num_alpha + molecule.num_beta
        num_spin_orbitals = molecule.num_orbitals * 2
        #Fermionic Operator
        ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
        #Remove list for reduced orbitals
        remove_list = self.oribital_reduction if self.oribital_reduction!= None else []
        remove_list = [x % molecule.num_orbitals for x in remove_list]
        #If we freeze the core orbitals then follow this list
        if self.freeze_core:
            freeze_list = [0] #Freeze 1s orbital
            freeze_list = [x % molecule.num_orbitals for x in freeze_list]

            remove_list = [x - len(freeze_list) for x in remove_list]
            remove_list += [x + molecule.num_orbitals - len(freeze_list)  for x in remove_list]

            freeze_list += [x + molecule.num_orbitals for x in freeze_list]
            ferOp, frozen_energy_shift = ferOp.fermion_mode_freezing(freeze_list)
        else:
            freeze_list = []
            #If we do not freeze the core orbitals then set frozen energy
            #shift to zero
            frozen_energy_shift = 0.0

        num_spin_orbitals -= len(freeze_list)
        num_particles -= len(freeze_list)
        #Eliminate high energy orbitals we do not need to simulate
        ferOp = ferOp.fermion_mode_elimination(remove_list)
        num_spin_orbitals -= len(remove_list)
        #Transform fermions into qubits
        qubitOp = ferOp.mapping(map_type=self.mapping, threshold=0.00000001)
        if self.two_qubit_reduction:
            # Reduce the M//2 and M th qubits if particle number and spin
            # are conserved
            # Only Valid for parity and H2 bravyi_kitaev
            qubitOp = Z2Symmetries.two_qubit_reduction(qubitOp, num_particles)
        self.shift = frozen_energy_shift + repulsion_energy
        #save the Hartree-Fock energy
        self.hf_energy = molecule.hf_energy
        #save the number of qubits needed
        self.num_qubits = qubitOp.num_qubits
        qubitOp.chop(1e-10)

        self.num_particles = num_particles
        self.num_orbitals = num_spin_orbitals
        return qubitOp.to_opflow()
