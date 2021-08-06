from ase.io import read
import numpy as np

class Environment:
    """
    Class representing part of structure which is not being altered via variation operators.
    I.e. it acts as environment for individual.
    """
    
    _DEFAULT_SUBSTRATE_SHIFT = 2.0

    def __init__(self, structure, offsetVector = None):
        """
        :param structure: atomic structure associated with this environment.
        :param offsetVector: vector to be added to molecules centers when assemble whole structure.
        If not provided such vector will be calculated on demand.
        """
        self._structure = structure
        self._offsetVector = offsetVector

    def calculateOffset(self, molecules, cell):
        """
        Calculate or retrieve vector to be added to each molecule when assemble whole structure.
        If such vector is not predefined for this environment it will be calculated basing on minimal atomic coordinates
        in nonperiodic direction.
        TODO working onli for 2D now.
        :param molecules: list of molecules for which the offset is being calculated.
        :param cell: TODO
        :return: offset vector.
        """
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
        """
        Retrieve atomic structure associated with environment.
        :return: atomic structure.
        """
        return self._structure


class World:
    """
    Class reprenting utility which generates possible environmemnts for calculation.
    """

    structureType = None
    atomType = None
    cellType = None

    @classmethod
    def registerTypes(cls, structureType, atomType, cellType):
        """
        Register types used by this utility.
        :param structureType: type representing atomic structure.
        :param atomType: type representing chemical element.
        :param cellType: type representing unit cell.
        """
        cls.structureType = structureType
        cls.atomType = atomType
        cls.cellType = cellType

    def __init__(self, files = None, pbc = (1,1,1)):
        """
        :param files: files with structures for possile environments.
        :param pbc: periodic boundary conditions of environment structures.
        """
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

    def putEnvironment(self, system):
        """
        Put environment in dictionary representing system.
        :param system: system dictionary.
        """
        if self._structures:
            structure = np.random.choice(self._structures)
            structure = structure.makeSupercell(np.round(structure.getCell().decomposeCell(system['cell'])))
            system['environment'] = Environment(structure)
