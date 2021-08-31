# Physics Models

This directory contains information about the physical systems simulated in our paper.

`PhysicsModel` is the parent class to all other physical systems


## Quantum Chemistry

`ElectronicStructureMolecule`: Is the parent class of all `QuantumChemistry` electronic structure systems that builds upon the characteristics of the PhysicsModel class. It uses the PySCFDriver to obtain information about the molecular systems.

`QuantumChemistryMolecules`: Here, the ElectronicStructureMolecule is initialized for the specific molecules studied (i.e, H2 and LiH)

## Lattice Schwinger Model

`LatticeSchwinger` constructs the physical system for the Lattice Schwinger Model
