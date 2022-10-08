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

    def getPrincipalAxes(self):
        """
        :return: 3x3 matrix of principal axes, main axes of inertia tensor (with all atom masses set to be equal).
        """
        coordinates = self._coordinates - self._coordinates.mean(axis=0)
        return np.linalg.eigh(np.eye(3) * np.sum(coordinates ** 2) - np.dot(coordinates.T, coordinates))

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

        cell = self.getCell()
        cellVectors = cell.getCellVectors()
        pbc = np.array(cell.getPBC(), dtype=bool) if cell is not None else np.array([False] * 3)
        dim = sum(pbc)

        whichPeriodic = np.where(pbc)[0]
        periodicVecs = cellVectors[pbc]
        nonperiodicVecs = cellVectors[~pbc]

        if dim == 0:
            vectors = self.getPrincipalAxes()[1].T
        elif dim == 1:
            periodicUnit = periodicVecs[0] / np.linalg.norm(periodicVecs[0])
            orthogPancake = self._coordinates - \
                               np.dot(self._coordinates, periodicUnit).reshape(-1, 1) * periodicUnit
            val, vectors = AtomicStructure(self._atomTypes, orthogPancake).getPrincipalAxes()
            vectors = vectors.T
            if val[0] < 1e-5:  # Check if inertia tensor has a singular matrix
                if np.dot(vectors[0], periodicUnit) == 1:
                    vectors[0] = vectors[1]
                vectors[0] -= np.dot(vectors[0], periodicUnit) * periodicUnit
                vectors[0] /= np.linalg.norm(vectors[0])
                vectors[1] = np.cross(periodicUnit, vectors[0])
            vectors[-1] = periodicVecs[0] # any 2D shape has a maximum inertia moment corresponding to orth direction
            vectors = np.roll(vectors, whichPeriodic[0] - 2, axis=0)
        elif dim == 2:
            normalvector = np.cross(periodicVecs[0], periodicVecs[1])
            normalvector *= np.sign(np.dot(normalvector, nonperiodicVecs[0]))
            vectors = cellVectors
            vectors[~pbc] = normalvector
        elif dim == 3:
            return cell
        else:
            raise ValueError(f'Incorrect dim: {dim}')

        newCell = type(cell)(vectors, pbc)
        return newCell

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


    def __init__(self, molecules=None, environment=None, pbc=(1,1,1)):
        """

        :param indices:
        :param environment:

        """
        self.indices = molecules
        self.environment = environment
        self.pbc = pbc
        if self.environment is not None:
            molIndices = set(np.concatenate(self.indices)) if self.indices else set()
            allIndices = list(range(len(molIndices) + len(self.environment.getStructure())))
            self.envIndices = np.asarray(list(set(allIndices).difference(molIndices)), dtype=int)
        else:
            self.envIndices = np.empty(0, dtype=int)

    @staticmethod
    def assemble(molecules, cell, environment=None, vacuumSize=0, **kwargs): # lots of work with calcs
        """

        :param molecules:
        :param cell:
        :param environment:
        :param kwargs:

        """
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
            assembledCell = envStructure.getCell()
        else:
            assembledCell = cell
        if vacuumSize > 0:
            structure = AtomicStructure(atomTypes, coordinates, assembledCell)
            assembledCell = structure.getRectifiedCell().getEnvelopeCell(coordinates, vacuumSize)
            coordinates = assembledCell.center(structure.getCartesianCoordinates())
        return (AtomicStructure(atomTypes, coordinates, assembledCell),   # cell depending on whether we have env
                AtomicDisassembler(indices, environment, cell.getPBC()))  # cell of molecules

    @staticmethod
    def assembleWithStyle(system, style, vacuumSize):
        if style == 'noEnvironment':
            structure, disassembler = AtomicDisassembler.assemble(system['molecules'], system['cell'],
                                                                  vacuumSize=vacuumSize)
            fixedIndices = []
        else:
            molecules, cell, environment = system['environment'].adjustSystem(system)
            if style in environment.processingStyles:
                structure, disassembler = getattr(environment, environment.processingStyles[style])(vacuumSize), None
                fixedIndices = environment.getFixedIndices()
            else:
                structure, disassembler = AtomicDisassembler.assemble(molecules, cell, environment,
                                                                      vacuumSize=vacuumSize)
                fixedIndices = disassembler.envIndices[environment.getFixedIndices()]
        system['disassembler'] = disassembler
        return structure, fixedIndices


    def disassemble(self, atomicStructure):
        """
        Decomposes given structure into molecules and environment.

        :param atomicStructure: structure to decompose.

        :return: {'molecules': <list of molecules>, 'cell': <Cell object>, 'environment': <optional environment object>}
        """
        atomTypes = atomicStructure.getAtomTypes()
        coordinates = atomicStructure.getCartesianCoordinates()
        if self.indices is None:
            syscoords = coordinates
            sysAtomTypes = atomTypes
        else:
            syscoords = []
            sysAtomTypes = []
            for indices in self.indices:
                syscoords.extend(coordinates[indices])
                sysAtomTypes.extend(atomTypes[indices])
            syscoords = np.array(syscoords)
            sysAtomTypes = np.asarray(sysAtomTypes)
        assembledCell = atomicStructure.getCell()
        cell = type(assembledCell)(assembledCell.getCellVectors(), pbc=self.pbc).getEnvelopeCell(syscoords, vacuumSize=1.0)
        system = dict()
        if self.environment is not None:
            envStructure = AtomicStructure(atomTypes[self.envIndices], coordinates[self.envIndices], assembledCell)
            system['environment'] = self.environment.getUpdatedEnvironment(envStructure)
        molecules = []
        indices = self.indices if self.indices is not None else np.arange(len(coordinates)).reshape((-1, 1))
        for inds in indices:
            molecules.append(AtomicStructure(atomTypes[inds], coordinates[inds]))
        system.update({'molecules': molecules, 'cell': cell})
        return system

    @staticmethod
    def udateSystemWithStyle(system, atomicStructure, style):
        disassembler = system.pop('disassembler')

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
