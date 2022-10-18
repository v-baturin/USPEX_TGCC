"""
USPEX.Atomistic.AtomicPrimitives
================================
"""

import numpy as np
from copy import copy
from collections import Counter

from .Transformation import Transformation


class AtomicStructure:
    """
    This class describe pure geometry of any atomic structure including crystals, molecules, nanoparticles etc.
    Such pure geometry consists of coordinates of atoms and their types. In case of periodic structures it naturally
    includes cell object which describes periodicity.
    """

    def __init__(self, atomTypes, coordinates, cell = None, zmatrixConfig = None):
        """

        :param atomTypes: list of types of atoms in structure. There are no no specific requirements to the data type of
            each atom type. It could even be None.
            Exception are **getComposition** and **getFormula** methods which require operator < to be defined.
        :param coordinates: array of coordinates of atoms. Its first dimension must coincide with size of atomTypes.
        :param cell: optional **Cell** object for periodic structures.
            Is expected to provide **getCellVectors** and **getPBC** methods.
        :param zmatrixConfig: TODO remove

        """
        assert len(atomTypes) == len(coordinates)
        self._atomTypes = np.asarray(atomTypes)
        self._coordinates = np.asarray(coordinates)
        self._cell = copy(cell)
        self.zmatrixConfig = copy(zmatrixConfig)

    @staticmethod
    def initFromFractionalCoordinates(atomTypes, coordinates, cell, zmatrixConfig = None):
        """
        Alternative constructor. Calculates cartesian coordinates from fractional coordinate and given **Cell** object.

        :param atomTypes: list of types of atoms in structure.
        :param coordinates: array of fractional coordinates of atoms. Its first dimension must coincide with size of atomTypes.
        :param cell: **Cell** object for periodic structures.
        :param zmatrixConfig: TODO remove

        """
        return AtomicStructure(atomTypes, cell.fractionalToCartesian(coordinates), cell, zmatrixConfig)

    def getAligned(self, axis):
        """
        Creates another **AtomicStructures** instance with the same cell parameters and atomic coordinates
         but aligned along given axis.

        :param axis: 3-vector along which the new structre will be aligned.
        :raises RuntimeError: if used on 0D or 3D structure.
        :return: new **AtomicStructures** instance.
        """
        return AtomicStructure.initFromFractionalCoordinates(self.getAtomTypes(), self.getFractionalCoordinates(),
                                                             self.getCell().getAlignedCell(axis))

    def __len__(self):
        return len(self._atomTypes)

    def getAtomTypes(self):
        """
        :return: copy of atom types sequence.
        """
        return copy(self._atomTypes)

    def getCell(self):
        """
        :return: copy of associated **Cell** object.
        """
        return copy(self._cell)

    def getZmatrixConfig(self):
        """
        TODO remove
        """
        return self.zmatrixConfig

    def getCartesianCoordinates(self):
        """
        :return: copy of cartesian coordinates of atoms in structure.
        """
        return copy(self._coordinates)

    def getFractionalCoordinates(self):
        """
        :return: calculates and returns fractional coordinates of atoms if structure has assosiated **Cell**  object.
        :raises: RuntimeError if structure does not have associated **Cell** object.
        """
        if self._cell is not None:
            return self._cell.cartesianToFractional(self._coordinates)
        else:
            raise RuntimeError("Call for fractional coordinates when cell is not set up.")

    def getComposition(self):
        """
        :return: calculates and returns composition of the structure as **Counter** object.
        """
        return Counter(dict(zip(*np.unique(self._atomTypes, return_counts=True))))

    def getFormula(self):
        """
        :return: calculates and returns formula of the structure.
        """
        return ''.join(f'{element}{amount}' for element, amount in self.getComposition().items())

    def getAllDistances(self):
        """
        :return: N*N matrix of pairwise distances between atoms, where N number of atoms in structure unit.
        """
        N = len(self._atomTypes)
        if N < 2:
            return np.zeros((N, N), dtype=float)
        from ase.geometry import get_distances
        cell = self._cell.getCellVectors() if self._cell is not None else None
        pbc = self._cell.getPBC() if self._cell is not None else None
        return get_distances(self._coordinates, cell = cell, pbc = pbc)[1]

    def getAllPairVectors(self):
        """
        :return: N*N matrix of pairwise vectors between atoms, where N number of atoms in structure unit.
        """
        from ase.geometry import get_distances
        cell = self._cell.getCellVectors() if self._cell is not None else None
        pbc = self._cell.getPBC() if self._cell is not None else None
        return get_distances(self._coordinates, cell = cell, pbc = pbc)[0]

    def getCenterOfMassCartesianCoordinates(self):
        """
        :return: calculates cartesian coordinates of geometrical center of atoms of structure unit.
        """
        return np.mean(self.getCartesianCoordinates(), axis=0)

    def getCenterOfMassFractionalCoordinates(self):
        """
        :return: calculates fractional coordinates of geometrical center of atoms of structure unit.
        """
        return np.mean(self.getFractionalCoordinates(), axis=0)

    def getTrigonalizedCellStructure(self):
        if self._cell is None:
            raise RuntimeError("Cell is not defined.")
        else:
            return self._cell.getTrigonalizeTransform().transform(self)

    def getRectifiedCell(self):
        """
        :return: **Cell** object depending on dimensionality.

            0d: Cell made of unit principal eigenvectors

            1d: Keep periodic vector from original cell, The other two are perpendicular to it,
            directed along principal directions of
            a structure, flatten along periodic vector

            2d: Keep periodic vectors from original cell. The third is a unity vector perpendicular to those two.

            3d: Returns original Cell
        """

        return self.getCell().getIntrinsicCell(self.getCartesianCoordinates())

    def makeSupercell(self, matrix):
        """
        For periodic structures constructs supercell representation of the same structure.
        I.e. it has associated **Cell** object multiple of initial **Cell** object
        and concatenated arrays of coordinates and atom types from blocks constituting the supercell.
        New cell vectros are given by formula

        >>> newCellVectors = matrix.dot(self.getCell().getCellVectors())

        :param matrix: 3*3 array of integers defining the supercell.

        :return: new **AtomicStructure** object.
        """
        assert not self.zmatrixConfig  # still not configured for these attributes
        atomTypes = copy(self._atomTypes)
        coordinates = copy(self._coordinates)
        cell = copy(self._cell)
        matrix = np.asarray(matrix, dtype=np.int16)
        if matrix.ndim == 1:
            matrix = np.array(matrix * np.eye(3), dtype=np.int16)

        newCell = type(cell)(matrix.dot(cell.getCellVectors()), cell.getPBC())
        latticePoints = _lattice_points_in_supercell(matrix).dot(newCell.getCellVectors())
        newCoordinates = []
        newAtomTypes = []
        for coord, atomType in zip(coordinates, atomTypes):
            newCoordinates.extend(latticePoints + coord)
            newAtomTypes.extend(np.repeat(atomType, len(latticePoints)))

        newCoordinates = newCell.getWrapedCartesianCoordinates(np.array(newCoordinates))
        newStructure = AtomicStructure(newAtomTypes, newCoordinates, newCell)
        return newStructure

    def getPerturbatedStructure(self, fixedIndices):
        # TODO don't perturbate fixed atoms
        coordinates = self._coordinates + 0.1 * (np.random.rand(len(self._coordinates), 3) - 0.5)
        return AtomicStructure(self._atomTypes, coordinates, self._cell)

class AtomicDisassembler:
    """
    This class describes how to assemble AtomicStructure from molecules and environment
    and then disassemble it back into molecules and environment. 
    """

    def __init__(self, molecules, environment=None, pbc=(1, 1, 1)):
        """

        :param indices:
        :param environment:

        """
        self.indices = molecules
        self.environment = environment
        self.pbc = pbc
        self.sysIndices = np.concatenate(self.indices)
        if self.environment is not None:
            allIndices = list(range(len(self.sysIndices) + len(self.environment.getStructure())))
            self.envIndices = np.asarray(list(set(allIndices).difference(set(self.sysIndices))), dtype=int)
            self.fixedIndices = self.envIndices[environment.getFixedIndices()]
        else:
            self.envIndices = np.empty(0, dtype=int)
            self.fixedIndices = np.empty(0, dtype=int)

    @staticmethod
    def assemble(molecules, cell, environment=None, vacuumSize=0, style=None, inStyle=None, **kwargs):
        """

        :param molecules:
        :param cell:
        :param environment:
        :param kwargs:

        """
        if f'{inStyle}.system' in kwargs:
            system = kwargs[f'{inStyle}.system']
            molecules, cell = system['molecules'], system['cell']
            environment = system['environment'] if 'environment' in system else None
        if style == 'noEnvironment':
            environment = None
        elif style == 'adjust':
            molecules, cell, environment = environment.adjustSystem(molecules, cell)
        elif environment is not None and style in environment.processingStyles:
            # TODO do we ever need to disassemble such structures?
            return getattr(environment, environment.processingStyles[style])(vacuumSize), None
        elif style is not None:
            raise ValueError(f"Style {style} is not valid.")
        pbc = cell.getPBC()
        atomTypes = []
        coordinates = []
        indices = []
        lowerBound = 0
        for molecule in molecules:
            atomTypes.extend(molecule.getAtomTypes())
            coordinates.extend(molecule.getCartesianCoordinates())
            size = len(molecule)
            indices.append(list(range(lowerBound, lowerBound + size)))
            lowerBound += size
        if environment is not None:
            envStructure = environment.getStructure()
            atomTypes.extend(envStructure.getAtomTypes())
            coordinates.extend(envStructure.getCartesianCoordinates())
            cell = envStructure.getCell()
        if vacuumSize > 0:
            cell = cell.getEnvelopeCell(coordinates, vacuumSize, intrinsic=True)
            coordinates = cell.center(coordinates)
        return AtomicStructure(atomTypes, coordinates, cell), AtomicDisassembler(indices, environment, pbc)

    def disassemble(self, structure):
        """
        Decomposes given structure into molecules and environment.

        :param structure: structure to decompose.

        :return: {'molecules': <list of molecules>, 'cell': <Cell object>, 'environment': <optional environment object>}
        """
        atomTypes = structure.getAtomTypes()
        coordinates = structure.getCartesianCoordinates()
        cell = structure.getCell()
        system = dict(
            molecules=[AtomicStructure(atomTypes[inds], coordinates[inds]) for inds in self.indices],
            cell=type(cell)(cell.getCellVectors(), pbc=self.pbc).getEnvelopeCell(coordinates[self.sysIndices],
                                                                                 vacuumSize=1.0)
        )
        if self.environment is not None:
            envStructure = AtomicStructure(atomTypes[self.envIndices], coordinates[self.envIndices], cell)
            system['environment'] = self.environment.getUpdatedEnvironment(envStructure)
        return system

    def decomposeDisplacements(self, displacements, structure):
        """
        Decompose atomic displacements into molecular transformations (translations and rotations)
        and atomic displacements relative to corresponding molecules.

        :type displacements: numpy array N*3
        :param displacements: array of atomic displacements, where N is number of atoms in structure.
        :param structure:

        :rtype: List[Tuple[**Transformation**, array of vectors]]
        :return: List of tuples for each molecule with transformation of the whole molecule
        and array of relative atomic displacements.
        """
        assert len(displacements) == len(structure)
        molecularDispacements = []
        molecules = self.disassemble(structure)['molecules']
        for molecule, inds in zip(molecules, self.indices):
            if len(molecule) > 1:
                atomicDisplacements = displacements[inds]
                centerCoordinates = molecule.getCenterOfMassCartesianCoordinates()
                atomicCoordinates = molecule.getCartesianCoordinates() - centerCoordinates
                inertia = np.linalg.norm(atomicCoordinates) ** 2
                atomicDistances = np.linalg.norm(atomicCoordinates, axis=1)
                nonCentralAtoms = np.nonzero(atomicDistances > 0.001)
                centralAtoms = np.nonzero(atomicDistances <= 0.001)
                atomicCoordinatesNonCentral = atomicCoordinates[nonCentralAtoms]
                atomicDistancesNonCentral = atomicDistances[nonCentralAtoms]
                atomicDisplacementsNonCentral = atomicDisplacements[nonCentralAtoms]
                atomicNormalsNonCentral = atomicCoordinatesNonCentral / atomicDistancesNonCentral.reshape((-1,1))
                translation = (np.sum(np.sum(atomicDisplacementsNonCentral * atomicNormalsNonCentral, axis=1).reshape((-1,1))
                                      * atomicNormalsNonCentral, axis=0) +
                               np.sum(atomicDisplacements[centralAtoms], axis=0)) / len(inds)
                rotation = np.sum(np.cross(atomicDisplacements, atomicCoordinates), axis=0) / inertia
                atomicDisplacements -= translation.reshape((1,3)) + np.cross(atomicCoordinates, rotation.reshape((1,3)))
            else:
                translation = displacements[inds]
                rotation = np.array([0., 0., 0.])
                atomicDisplacements = np.array([[0.,0.,0.]])
            molecularDispacements.append((Transformation.fromRotVector(rotation, translation), atomicDisplacements))
        return molecularDispacements

    def findMolIndex(self, index):
        for molIndex, inds in enumerate(self.indices):
            if index in inds:
                return molIndex
        raise RuntimeError("Bad index or empty structure.")


def _lattice_points_in_supercell(supercell_matrix):
    """
    Returns the list of points on the original lattice contained in the
    supercell in fractional coordinates (with the supercell basis).
    e.g. [[2,0,0],[0,1,0],[0,0,1]] returns [[0,0,0],[0.5,0,0]]

    Args:
        supercell_matrix: 3x3 matrix describing the supercell

    Returns:
        numpy array of the fractional coordinates
    """
    diagonals = np.array(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1],
         [1, 1, 0], [1, 1, 1]])
    d_points = np.dot(diagonals, supercell_matrix)

    mins = np.min(d_points, axis=0)
    maxes = np.max(d_points, axis=0) + 1

    ar = np.arange(mins[0], maxes[0])[:, None] * np.array([1, 0, 0])[None, :]
    br = np.arange(mins[1], maxes[1])[:, None] * np.array([0, 1, 0])[None, :]
    cr = np.arange(mins[2], maxes[2])[:, None] * np.array([0, 0, 1])[None, :]

    all_points = ar[:, None, None] + br[None, :, None] + cr[None, None, :]
    all_points = all_points.reshape((-1, 3))

    frac_points = np.dot(all_points, np.linalg.inv(supercell_matrix))

    tvects = frac_points[np.all(frac_points < 1 - 1e-10, axis=1)
                         & np.all(frac_points >= -1e-10, axis=1)]
    assert len(tvects) == round(abs(np.linalg.det(supercell_matrix)))
    return tvects
