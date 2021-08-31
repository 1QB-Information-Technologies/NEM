from physics_models.ElectronicStructureMolecule import ElectronicStructureMolecule

class H2molecule(ElectronicStructureMolecule):
    '''
    Construct the PhysicsModel for the electronic structure of H2
    '''
    def __init__(self,
                    bond_length: float,
                    mapping: ['parity', 'jordan_wigner', 'bravyi_kitaev'] = 'bravyi_kitaev'):
        '''
        Initialize the H2molecule class which can be used for VQE, VMC, ED

        Parameters
        -------------
        bond_length = float
            Specifies the bond length (angstrom) between the atoms
        mapping = str
            one of: ['parity', 'jordan_wigner', 'bravyi_kitaev']
            specifies the mapping between fermionic operators in second
            quantization to qubit operators. Default "bravyi_kitaev" to
            reproduce results from "Neural Error Mitigation of Near-term
            Quantum Simulations"
        ham_type = str
            one of: ['Old', 'New']
            Tells the ElectronicStructureMolecule to construct the qubit
            hamiltonian for the molecule using the 'new' method or the 'old' one.
        '''
        #Molecule Information
        self.bond_length = bond_length
        atom_string = "H .0 .0 .0; H .0 .0 " + str(self.bond_length)
        orbital_reduction = None
        freeze_core = False
        two_qubit_reduction = True
        # Construct hamiltonian using "Old" method in ElectronicStructureMolecule
        ham_type = "Old"
        if ham_type == "New" and mapping == 'bravyi_kitaev' and two_qubit_reduction == True:
            raise ValueError("New ham_type method does not do two_qubit_reuction" +
                            "for 'bravyi_kitaev' mapping, only 'parity'" +
                            "\n     Please use 'Old' method")
                            
        super().__init__(   name = "H2molecule",
                            atom = atom_string,
                            mapping = mapping,
                            ham_type = ham_type,
                            orbital_reduction = orbital_reduction,
                            freeze_core = freeze_core,
                            two_qubit_reduction = two_qubit_reduction)

class LiHmolecule(ElectronicStructureMolecule):
    '''
    Construct the PhysicsModel for the electronic structure of LiH
    '''
    def __init__(self,
                bond_length: float,
                mapping: ['parity', 'jordan_wigner', 'bravyi_kitaev'] = 'parity'):
        '''
        Initialize the LiH_molecule class which can be used for VQE, VMC, ED

        Parameters
        -------------
        bond_length = float
            Specifies the bond length (angstrom) between the atoms
        mapping = str
            one of: ['parity', 'jordan_wigner', 'bravyi_kitaev']
            specifies the mapping between fermionic operators in second
            quantization to qubit operators. Default 'parity', what we use
            in "Neural Error Mitigation of Near-term Quantum Simulations"
        '''
        # Molecule Information

        self.bond_length = bond_length
        atom_string = "Li .0 .0 .0; H .0 .0 " + str(self.bond_length)
        orbital_reduction = [-3,-2]
        freeze_core = True
        two_qubit_reduction = True
        # Construct hamiltonian using "New" method in ElectronicStructureMolecule
        ham_type = "New"
        super().__init__(   name =  "LiHmolecule",
                            atom = atom_string,
                            mapping = mapping,
                            ham_type = ham_type,
                            orbital_reduction = orbital_reduction,
                            freeze_core = freeze_core,
                            two_qubit_reduction = two_qubit_reduction)
