import numpy as np
import yaml
from pathlib import Path
from copy import copy

from ..Semantics.Atomistic.Atomistic import Atomistic as AtomisticSemantics
from ..Expressions.Functions.AtomisticFunctions import AtomisticFunctions
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
        edges = []
        lowerBound = 0
        for molecule in molecules:
            atomTypes.extend(molecule.getAtomTypes())
            coordinates.extend(molecule.getCartesianCoordinates())
            size = len(molecule)
            inds = np.arange(lowerBound, lowerBound + size)
            indices.append(inds)
            for edge in molecule.edges:
                edges.append(inds[edge])
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
        return (Atomistic.structureType(atomTypes, coordinates, cell, edges=edges),
                AtomicDisassembler(indices, envIndices, fixedIndices, pbc))

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
        molecules = []
        for inds in self.indices:
            edges = []
            for i, j in structure.edges:
                ii = np.where(inds == i)[0]
                jj = np.where(inds == j)[0]
                if len(ii) == 1 and len(jj) == 1:
                    edges.append((ii[0], jj[0]))
            molecules.append(Atomistic.structureType(atomTypes[inds], coordinates[inds] + offset, edges=edges))
        environments = []
        for eInds, fInds in zip(self.envIndices, self.fixedIndices):
            envStructure = Atomistic.structureType(atomTypes[eInds], coordinates[eInds] + offset, cell)
            indices = np.argwhere(eInds.reshape((-1, 1)) == fInds.reshape((1, -1)))[:, 0]
            environments.append((envStructure, indices))
        return {'atomistic.molecules': molecules, 'atomistic.cell': sysCell, 'atomistic.environments': environments}

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


class Atomistic(AtomisticSemantics):
    structureType = None
    atomType = None
    cellType = None
    AtomicStructureRepresentation = None
    atomicDisassemblerType = AtomicDisassembler

    @classmethod
    def registerTypes(cls, structureType, atomType, cellType, AtomicStructureRepresentation):
        cls.structureType = structureType
        cls.atomType = atomType
        cls.cellType = cellType
        cls.AtomicStructureRepresentation = AtomicStructureRepresentation
        AtomicStructureRepresentation.registerTypes(structureType, atomType, cellType)

    @classmethod
    def writeAtomicStructure(cls, filename, system: dict):
        filename = Path(filename)
        cls.writeAtomicStructures(filename, [system])

    @classmethod
    def writeAtomicStructures(cls, filename: Path, systems: list):
        structures = []
        labels = []
        descriptions = []
        printUSPEX = False
        for i, system in enumerate(systems):
            try:
                structure = system['atomistic.structure'] # vacuumSize=10.0
            except RecursionError:
                continue
            disassembler = system['atomistic.disassembler']
            atomTypes = structure.getAtomTypes()
            coordinates = structure.getCartesianCoordinates()
            sortIndices = np.argsort(atomTypes)
            reversedIndices = np.argsort(sortIndices)
            structure = cls.structureType(atomTypes[sortIndices], coordinates[sortIndices], structure.getCell())
            structures.append(structure)
            labels.append(system['.label'])
            d = {'filename': filename.name, 'index': i}
            pbc = system['atomistic.cell'].getPBC()
            if pbc != (1, 1, 1):
                d['pbc'] = ' '.join(f'{c}' for c in pbc)
                printUSPEX = True
            molecules = []
            for indices in disassembler.indices:
                if len(indices) > 1:
                    molecules.append(' '.join(f'{ind}' for ind in reversedIndices[indices]))
                    printUSPEX = True
            if molecules:
                d['molecules'] = molecules
            if 'atomistic.environments' in system and len(system['atomistic.environments']):
                printUSPEX = True
                d['environments'] = []
                for eInds in disassembler.envIndices:
                    d['environments'].append(' '.join(f'{ind}' for ind in eInds))
                d['fixed'] = ' '.join(f'{ind}' for ind in disassembler.allFixedIndices)
            descriptions.append(d)
        cls.AtomicStructureRepresentation.writePOSCARS(filename, structures, labels)
        if printUSPEX:
            with open(f'{filename}.uspex', 'wt') as f:
                f.write(yaml.safe_dump(descriptions))

    @classmethod
    def readAtomicStructure(cls, filename) -> dict:
        return cls.readAtomicStructures(filename)[0]

    @classmethod
    def readAtomicStructures(cls, filename) -> list:
        filename = Path(filename)
        directory = filename.parent
        if filename.suffix == '.uspex':
            with open(filename) as f:
                descriptions = yaml.safe_load(f.read())
            files = {name: cls.AtomicStructureRepresentation.readPOSCARS(directory/name)
                     for name in np.unique([s['filename'] for s in descriptions])}
            systems = []
            for d in descriptions:
                d = copy(d)
                structure = files[d.pop('filename')][d.pop('index')]
                allIndSet = set(range(len(structure)))
                d['indices'] = []
                if 'pbc' in d:
                    d['pbc'] = tuple(int(c) for c in d.pop('pbc').split(' '))
                if 'molecules' in d:
                    d['indices'] = [np.array(mol.split(' '), dtype=int) for mol in d.pop('molecules')]
                    molIndSet = set(np.concatenate(d['indices']))
                else:
                    molIndSet = set()
                fixed = np.array(d.pop('fixed').split(' '), dtype=int) if 'fixed' in d else np.empty(0, dtype=int)
                if 'environments' in d:
                    d['envIndices'] = []
                    d['fixedIndices'] = []
                    for eInds in d.pop('environments'):
                        eInds = np.array(eInds.split(' '), dtype=int)
                        fInds = np.argwhere(eInds.reshape((-1, 1)) == fixed.reshape((1, -1)))[:, 0]
                        d['envIndices'].append(eInds)
                        d['fixedIndices'].append(fInds)
                    envIndSet = set(np.concatenate(d['envIndices']))
                else:
                    envIndSet = set()
                d['indices'].extend(np.fromiter(allIndSet - envIndSet - molIndSet, dtype=int).reshape((-1, 1)))
                systems.append(cls.atomicDisassemblerType(**d).disassemble(structure))
        else:
            systems = [cls.atomicDisassemblerType(np.arange(len(structure)).reshape((-1, 1))).disassemble(structure)
                       for structure in cls.AtomicStructureRepresentation.readPOSCARS(filename)]
        return systems

    def propertyExtension(self):
        return AtomisticFunctions(self)

