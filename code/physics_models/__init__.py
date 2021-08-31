"""
The classes here represent different quantum qubit systems, chiefly by providing methods to access their Hamiltonians.
The parent class is PhysicsModel, its docstring documents the common methods / interface. In addition to the common
methods, some classes implement additional things you might be interested in specific to the model, such as methods to
calculate the order parameter.
"""


from .PhysicsModel import PhysicsModel
from .LatticeSchwinger import LatticeSchwinger
from .QuantumChemistryMolecules import H2molecule, LiHmolecule
