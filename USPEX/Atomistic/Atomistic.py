import numpy as np

from USPEX.Expressions.Functions.AtomisticFunctions import AtomisticFunctions
from .Transformation import Transformation


class AtomicDisassembler:
    """
    This class describes how to assemble AtomicStructure from molecules and environment
    and then disassemble it back into molecules and environment.
    """

    def __init__(self, indices, envIndices=None, fixedIndices=None, pbc=(1, 1, 1)):
        """

        :param indices:
        :param environment:

        """
        self.indices = indices
        self.pbc = pbc
        self.sysIndices = np.concatenate(self.indices)
        self.envIndices = []
        self.fixedIndices = []
        if envIndices is not None:
            for eInds, fInds in zip(envIndices, fixedIndices):
                eInds = np.asarray(eInds, dtype=int)
                fInds = np.asarray(fInds, dtype=int)
                self.envIndices.append(eInds)
                self.fixedIndices.append(eInds[fInds])
        self.allFixedIndices = np.concatenate(self.fixedIndices) if self.fixedIndices else np.empty(0, dtype=int)

    @staticmethod
    def assemble(system, **kwargs):
        """

        :param molecules:
        :param cell:
        :param environments:
        :param kwargs:

        """
        molecules = system['atomistic.molecules']
        cell = system['atomistic.cell']
        environments = system['atomistic.environments'] if 'atomistic.environments' in system else None
        vacuumSize = system['.vacuumSize'] if '.vacuumSize' in system else 0
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
        envIndices = []
        fixedIndices = []
        if environments is not None:
            for envStructure, inds in environments:
                atomTypes.extend(envStructure.getAtomTypes())
                coordinates.extend(envStructure.getCartesianCoordinates())
                cell = envStructure.getCell()
                size = len(envStructure)
                envIndices.append(list(range(lowerBound, lowerBound + size)))
                lowerBound += size
                fixedIndices.append(inds)
        if vacuumSize > 0:
            cell = cell.getEnvelopeCell(coordinates, vacuumSize, intrinsic=True)
            coordinates = cell.center(coordinates)
        return Atomistic.structureType(atomTypes, coordinates, cell), AtomicDisassembler(indices, envIndices, fixedIndices, pbc)

    def disassemble(self, structure):
        """
        Decomposes given structure into molecules and environment.

        :param structure: structure to decompose.

        :return: {'molecules': <list of molecules>, 'cell': <Cell object>, 'environment': <optional environment object>}
        """
        atomTypes = structure.getAtomTypes()
        coordinates = structure.getCartesianCoordinates()
        cell = structure.getCell()
        sysCoordinates = coordinates[self.sysIndices]
        sysCell = type(cell)(cell.getCellVectors(), pbc=self.pbc).getEnvelopeCell(sysCoordinates, vacuumSize=1.0)
        offset = np.mean(sysCell.center(sysCoordinates) - sysCoordinates, axis=0)
        system = {
            'atomistic.molecules': [Atomistic.structureType(atomTypes[inds], coordinates[inds] + offset) for inds in
                                    self.indices], 'atomistic.cell': sysCell, 'atomistic.environments': []}
        for eInds, fInds in zip(self.envIndices, self.fixedIndices):
            envStructure = Atomistic.structureType(atomTypes[eInds], coordinates[eInds] + offset, cell)
            indices = np.argwhere(eInds.reshape((-1, 1)) == fInds.reshape((1, -1)))[:, 0]
            system['atomistic.environments'].append((envStructure, indices))
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
        molecules = self.disassemble(structure)['atomistic.molecules']
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


class Atomistic:
    structureType = None
    atomType = None
    cellType = None
    atomicDisassemblerType = AtomicDisassembler

    propertyExtension = AtomisticFunctions

    @classmethod
    def registerTypes(cls, structureType, atomType, cellType):
        cls.structureType = structureType
        cls.atomType = atomType
        cls.cellType = cellType
