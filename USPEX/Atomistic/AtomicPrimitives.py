import numpy as np
from copy import copy
from collections import Counter

from .Transformation import Transformation


class AtomicStructure:

    def __init__(self, atomTypes, coordinates, cell = None, bonds = None, zmatrixConfig = None, **kwargs):
        assert len(atomTypes) == len(coordinates)
        self.atomTypes = np.asarray(atomTypes)
        self.coordinates = np.asarray(coordinates)
        self.cell = copy(cell)
        self.bonds = copy(bonds)
        self.zmatrixConfig = copy(zmatrixConfig)

    def __len__(self):
        return len(self.atomTypes)

    def makeSupercell(self, matrix):
        assert not (self.bonds and self.zmatrixConfig) # still not configured for these attributes
        atomTypes = copy(self.atomTypes)
        coordinates = copy(self.coordinates)
        cell = copy(self.cell)
        matrix = np.asarray(matrix, dtype=np.int16)
        if matrix.ndim == 1:
            matrix = np.array(matrix*np.eye(3), dtype=np.int16)
        
        newCell = type(cell)(matrix.dot(cell.getCellVectors()), cell._pbc)
        latticePoints = lattice_points_in_supercell(matrix).dot(newCell.getCellVectors())
        newCoordinates = []
        newAtomTypes = []
        for coord, atomType in zip(coordinates, atomTypes):
            newCoordinates.extend(latticePoints + coord)
            newAtomTypes.extend(np.repeat(atomType, len(latticePoints)))
        
        newCoordinates = newCell.getWrapedCartesianCoordinates(np.array(newCoordinates))
        newStructure = AtomicStructure(newAtomTypes, newCoordinates, newCell)
        return newStructure

    def getAtomTypes(self):
        return copy(self.atomTypes)

    def getZmatrixConfig(self):
        return self.zmatrixConfig

    def getComposition(self):
        return Counter(dict(zip(*np.unique(self.atomTypes, return_counts=True))))

    def getFormula(self):
        return ''.join(f'{element.short_name}{amount}' for element, amount in self.getComposition().items())

    def getCartesianCoordinates(self):
        return copy(self.coordinates)

    def getFractionalCoordinates(self):
        if self.cell is not None:
            return self.cell.cartesianToFractional(self.coordinates)
        else:
            raise RuntimeError("Call for fractional coordinates when cell is not set up.")

    def getAllDistances(self):
        N = len(self.atomTypes)
        if N < 2:
            return np.zeros((N, N), dtype=float)
        from ase.geometry import get_distances
        cell = self.cell.getCellVectors() if self.cell is not None else None
        pbc = self.cell.getPBC() if self.cell is not None else None
        return get_distances(self.coordinates, cell = cell, pbc = pbc)[1]

    def getAllPairVectors(self):
        from ase.geometry import get_distances
        cell = self.cell.getCellVectors() if self.cell is not None else None
        pbc = self.cell.getPBC() if self.cell is not None else None
        return get_distances(self.coordinates, cell = cell, pbc = pbc)[0]

    def getCenterOfMassCartesianCoordinates(self):
        return np.mean(self.getCartesianCoordinates(), axis=0)

    def getCenterOfMassFractionalCoordinates(self):
        return np.mean(self.getFractionalCoordinates(), axis=0)

    def getPrincipalAxes(self, intactPBCVectors=False):
        """
        :rtype: 3x3 numpy array
        :return: principle axes, main axes of inertia tensor (with all atom masses set to be equal).
        """
        coordinates = self.coordinates - self.coordinates.mean(axis=0)
        return np.linalg.eigh(np.eye(3) * np.sum(coordinates ** 2) - np.dot(coordinates.T, coordinates))

    def getRectifiedCell(self):
        """
        returns Cell object for subsequent vacuum adding. The cellVectors are:
        for 0d: unit principal eigenvectors
        for 1d: Periodic vector remains, the other two are perpendicular to it, directed along principal directions of
        a structure, flatten along periodic vector
        for 2d: Periodic vectors remain. The third is a unity vector perpendicular to 2d system
        for 3d: Returns initial Cell
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
            orthogPancake = self.coordinates - \
                               np.dot(self.coordinates, periodicUnit).reshape(-1, 1) * periodicUnit
            vectors = AtomicStructure(self.atomTypes, orthogPancake).getPrincipalAxes()[1].T
            vectors[-1] = periodicVecs[0]
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

    def getCell(self):
        return copy(self.cell)

    @staticmethod
    def initFromFractionalCoordinates(atomTypes, coordinates, cell, **kwargs):
        return AtomicStructure(atomTypes, cell.fractionalToCartesian(coordinates), cell, **kwargs)

    @staticmethod
    def assemble(molecules, cell, environment=None, **kwargs):
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
            coordinates = list(np.asarray(coordinates, dtype = float) + environment.calculateOffset(molecules, cell))
            atomTypes.extend(environment.getStructure().getAtomTypes())
            coordinates.extend(environment.getStructure().getCartesianCoordinates())
        return (AtomicStructure(atomTypes, coordinates, cell, **kwargs),
                AtomicDisassembler(indices, environment))


class AtomicDisassembler:

    def __init__(self, indices, environment):
        self.indices = [np.asarray(inds, dtype = int) for inds in indices]
        self.environment = environment

    @property
    def envIndices(self):
        if self.environment is not None:
            molIndices = set(np.concatenate(self.indices))
            allIndices = list(range(len(molIndices) + len(self.environment.getStructure())))
            return np.asarray(list(set(allIndices).difference(molIndices)), dtype = int)
        else:
            return np.empty(0, dtype=int)

    @staticmethod
    def createFlatDisassembler(N):
        return AtomicDisassembler([[i] for i in range(N)], None)

    def disassemble(self, atomicStructure):
        atomTypes = atomicStructure.getAtomTypes()
        coordinates = atomicStructure.getCartesianCoordinates()
        cell = atomicStructure.getCell()
        molecules = []
        for indices in self.indices:
            molecules.append(AtomicStructure(atomTypes[indices], coordinates[indices]))
        system = {'molecules': molecules, 'cell': cell}
        if self.environment is not None:
            envStructure = AtomicStructure(atomTypes[self.envIndices], coordinates[self.envIndices], cell)
            system['environment'] = type(self.environment)(envStructure, offsetVector = np.zeros(3, dtype=float))
        return system

    def decomposeDisplacements(self, displacements, structure):
        """
        Decompose atomic displacements into molecular translations and rotations and intramolecular atomic displacements.
        :type displacements: numpy array N*3
        :param displacements: array of atomic displacements, where N is number of atoms in structure.
        :rtype: List[Tuple[vector, vector, array of vectors]]
        :return: List of tuples for each molecule with translation vector, rotation vector and array of intramolecular
        atomic displacements.
        """
        assert len(displacements) == len(structure)
        molecularDispacements = []
        molecules = self.disassemble(structure)['molecules']
        for molecule, inds in zip(molecules, self.indices):
            if len(molecule) > 1:
                atomicDisplacements = displacements[inds]
                centerCoordinates = molecule.coordinates.mean(axis=0)
                atomicCoordinates = molecule.coordinates - centerCoordinates
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

def lattice_points_in_supercell(supercell_matrix):
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
