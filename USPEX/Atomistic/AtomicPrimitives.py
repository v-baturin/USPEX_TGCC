import numpy as np
from copy import copy
from collections import Counter

from .Transformation import Transformation
from .CellUtility import Cell
from scipy.spatial.distance import cosine


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

    def getPrincipalCell(self):
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
            orthogPancake = self.coordinates - \
                               np.dot(self.coordinates, periodicVecs[0]).reshape(-1, 1) * \
                               periodicVecs[0] / np.sum(periodicVecs[0] ** 2)
            orthogPancake -= np.mean(orthogPancake, axis=0)
            vectors = AtomicStructure(self.atomTypes, orthogPancake).getPrincipalAxes()[1].T
            assert np.abs(cosine(vectors[-1], periodicVecs[0]) - 1) > 0.99  # TODO: remove, everything should go fine
            vectors[-1] = periodicVecs[0]
            vectors = np.roll(vectors, whichPeriodic[0] - 2, axis=0)
        elif dim == 2:
            normalvector = np.cross(periodicVecs[0], periodicVecs[1])
            normalvector *= np.sign(np.dot(normalvector, nonperiodicVecs[0]))
            vectors = cellVectors
            vectors[~pbc] = normalvector
        elif dim == 3:
            vectors = cellVectors
        else:
            raise ValueError(f'Incorrect dim {dim}')

        newCell = Cell(vectors, pbc)
        return newCell

    def getPrincipalTransformation(self):
        cell = self.getCell()
        pbc = cell.getPBC() if cell is not None else (0,0,0)
        dim = sum(pbc)
        if (dim == 3) or (dim == 2):
            transformation = Transformation(np.eye(3), np.zeros((3,)))
        elif dim == 0:
            values, vectors = self.getPrincipalAxes()
            center = self.coordinates.mean(axis=0)
            transformation = Transformation.fromMatrix(vectors, center - np.dot(vectors, center))
        elif dim == 1:
            # if toPrincipalAxes:
            #     forAxes = coordinates * affectedDims
            #     _, rotMatrix = np.linalg.eigh(np.eye(3) * np.sum(forAxes ** 2) - np.dot(forAxes.T, forAxes))
            #     coordinates = np.dot(coordinates, rotMatrix)
            transformation = None
        else:
            raise ValueError(f'Incorrect dim {dim}')
        return transformation

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

if __name__ == "__main__":
    from ase.io import read
    ase_ats = read('/home/vsbat/USPEX_PY2/material_mp-160_files/POSCAR.mp-160_B', format='vasp')
    print(ase_ats.cell)
    mycell = Cell(ase_ats.cell, pbc=(0,0,1))

