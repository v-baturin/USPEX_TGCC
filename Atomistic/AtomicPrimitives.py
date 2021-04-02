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

    def getAtomTypes(self):
        return copy(self.atomTypes)

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

    def getPrincipleAxes(self):
        """
        :rtype: 3x3 numpy array
        :return: principle axes, main axes of inertia tensor (with all atom masses set to be equal).
        """
        coordinates = self.coordinates - self.coordinates.mean(axis=0)
        inertia = np.zeros((3, 3), dtype=float)  # moment of inertia tensor
        inertia[0, 0] = (coordinates[:, 1] ** 2 + coordinates[:, 2] ** 2).sum()
        inertia[1, 1] = (coordinates[:, 0] ** 2 + coordinates[:, 2] ** 2).sum()
        inertia[2, 2] = (coordinates[:, 0] ** 2 + coordinates[:, 1] ** 2).sum()
        inertia[0, 1] = -(coordinates[:, 0] * coordinates[:, 1]).sum()
        inertia[1, 2] = -(coordinates[:, 1] * coordinates[:, 2]).sum()
        inertia[2, 0] = -(coordinates[:, 2] * coordinates[:, 0]).sum()
        inertia[1, 0] = -(coordinates[:, 0] * coordinates[:, 1]).sum()
        inertia[2, 1] = -(coordinates[:, 1] * coordinates[:, 2]).sum()
        inertia[0, 2] = -(coordinates[:, 2] * coordinates[:, 0]).sum()
        return np.linalg.eigh(inertia)

    def getCell(self):
        return copy(self.cell)

    @staticmethod
    def initFromFractionalCoordinates(atomTypes, coordinates, cell, **kwargs):
        return AtomicStructure(atomTypes, cell.fractionalToCartesian(coordinates), cell, **kwargs)

    @staticmethod
    def assemble(molecules, cell, environment = None, **kwargs):
        atomTypes = []
        coordinates = []
        indices = []
        lowerBound = 0
        for molecule in molecules:
            atomTypes.extend(molecule.atomTypes)
            coordinates.extend(molecule.coordinates)
            size = len(molecule)
            indices.append(list(range(lowerBound, lowerBound + size)))
            lowerBound += size
        # offsetVector = environment.calculateOffset(molecules, cell)
        structure = AtomicStructure(atomTypes, coordinates, cell, **kwargs)
        # structure.translate(offsetVector)
        coordinates = structure.getCartesianCoordinates()
        # atomTypes.extend(environment.getStructure().getAtomTypes())
        # coordinates.extend(environment.getStructure().getCortesianCoordinates())
        return (AtomicStructure(atomTypes, coordinates, cell, **kwargs),
                AtomicDisassembler(indices, environment))


class AtomicDisassembler:

    def __init__(self, indices, environment):
        self.indices = indices
        self.environment = copy(environment)

    @staticmethod
    def createFlatDisassembler(N):
        return AtomicDisassembler([[i] for i in range(N)], None)

    def disassemble(self, atomicStructure):
        atomTypes = list(atomicStructure.getAtomTypes())
        coordinates = list(atomicStructure.getCartesianCoordinates())
        molecules = []
        for indices in self.indices:
            molecules.append(AtomicStructure([atomTypes[i] for i in indices], [coordinates[i] for i in indices]))
        # assert len(atomTypesNotYet) == len(coordinatesNotYet)
        # assert len(coordinatesNotYet) == 0 # len(self.environment.getStructure())
        return {'molecules': molecules, 'cell': atomicStructure.getCell(), 'environment': copy(self.environment)}

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
