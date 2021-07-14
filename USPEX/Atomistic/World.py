from ase.io import read
import numpy as np

class Environment:
    
    _DEFAULT_SUBSTRATE_SHIFT = 2.0

    def __init__(self, structure, offsetVector = None):
        self._structure = structure
        self._offsetVector = offsetVector

    def calculateOffset(self, molecules, cell):
        if self._offsetVector is not None:
            offsetVector = self._offsetVector
        else:
            cell = self._structure.getCell()
            pbc = cell.getPBC()
            assert sum(pbc) == 2 # For substrates only
            zeroPBC = np.where(np.array(pbc)==0)[0]
            axis = cell.getCellVectors()[zeroPBC]
            envCoordinates = cell.cartesianToFractional(self._structure.getCartesianCoordinates())[:,zeroPBC]
            coordinates = np.vstack([cell.cartesianToFractional(molecule.getCartesianCoordinates())
                                     for molecule in molecules])[:,zeroPBC]
            offsetVector = axis * (self._DEFAULT_SUBSTRATE_SHIFT / np.linalg.norm(axis)
                                   + envCoordinates.max() - coordinates.min())
        return offsetVector
        
    def getStructure(self):
        return self._structure


class World:

    structureType = None
    atomType = None
    cellType = None

    def __init__(self, files = None, pbc = (1,1,1)):
        self._structures = []
        self._pbc = pbc
        self.files = files if files is not None else []
        for file in self.files:
            atoms = read(file)
            atomTypes = [self.atomType(item) for item in atoms.symbols]
            coordinates = atoms.get_positions()
            cell = self.cellType(atoms.get_cell().array, self._pbc).getEnvelopeCell(coordinates)
            coordinates = cell.center(coordinates)
            self._structures.append(self.structureType(atomTypes, coordinates, cell))

    @classmethod
    def registerTypes(cls, structureType, atomType, cellType):
        cls.structureType = structureType
        cls.atomType = atomType
        cls.cellType = cellType

    def putEnvironment(self, system):
        if self._structures:
            structure = np.random.choice(self._structures)
            structure = structure.makeSupercell(np.round(structure.getCell().decomposeCell(system['cell'])))
            system['environment'] = Environment(structure)
