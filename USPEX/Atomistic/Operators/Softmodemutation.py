import logging
logger = logging.getLogger(__name__)

import numpy as np

from ..Transformation import Transformation
from ...Atomistic.softmodes.calcSoftModes import calcSoftModes


_MIN_VALID_FREQUENCY = 5.0e-4


class Softmodemutation:
    def __init__(self, utilities, degree: float = None):
        self.simpleMoleculeUtility = utilities.simpleMoleculeUtility
        self.ionDistances = utilities.ionDistances
        self.bonds = utilities.bonds
        self.environmentUtility = utilities.environmentUtility
        self.conditions = utilities.conditions
        self.cellUtility = utilities.cellUtility
        self.degree= degree
        self.knownSystems = {}

    def __call__(self, system):
        ID = system['ID']
        molecules = system['molecules']
        cell = system['cell']
        environment = system['environment'] if 'environment' in system else None
        structure, disassembler = self.simpleMoleculeUtility.atomicDisassemblerType.assemble(molecules, cell)
        if self.cellUtility.isGoodCell(cell.getEnvelopeCell(structure.getCartesianCoordinates())):
            degree = self.degree if self.degree else np.mean([el.covalent_radius for el in structure.getAtomTypes()]) * 3
            if ID in self.knownSystems:
                frequencies, eigenVectors = self.knownSystems[ID]
            else:
                try:
                    bonds = self.bonds.getMinimalGraphBonds(structure)
                except Exception as e:
                    logger.debug(e)
                    raise RuntimeError("Softmutation failed.")
                frequencies, eigenVectors = calcSoftModes(structure, bonds)
                self.knownSystems[ID] = (frequencies, eigenVectors)
            while len(frequencies) > 0:
                freq = frequencies.pop(0)
                eigenVector = eigenVectors.pop(0)
                if freq < _MIN_VALID_FREQUENCY:
                    continue
                displacements = eigenVector.reshape((len(structure),3))
                displacements *= degree/np.max(np.linalg.norm(displacements, axis = 1))
                molecules1 = []
                molecules2 = []
                for (transformation, _), molecule in zip(disassembler.decomposeDisplacements(displacements, structure),
                                                                                   molecules):

                    offset = Transformation.fromRotVector([0.,0.,0.,], molecule.getCenterOfMassCartesianCoordinates())
                    molecule = (-offset).transform(molecule)
                    molecule1 = transformation.transform(molecule)
                    molecule2 = (-transformation).transform(molecule)
                    molecule1 = offset.transform(molecule1)
                    molecule2 = offset.transform(molecule2)
                    molecules1.append(molecule1)
                    molecules2.append(molecule2)

                offsprings = ()
                system = {'molecules': molecules1, 'cell': cell}
                self.environmentUtility.putEnvironment(system, environment)
                atomSymbols, atomDistances, disassembler1 = self.simpleMoleculeUtility.getMinDistances(**system)
                minDistMatrix = self.ionDistances.getDistances(atomSymbols, self.conditions.externalPressure)
                if disassembler1.environment is not None:
                    inds = disassembler1.envIndices
                    atomDistances[tuple(np.meshgrid(inds, inds))] = minDistMatrix[tuple(np.meshgrid(inds, inds))]
                if np.all(atomDistances >= minDistMatrix):
                    self.conditions.putConditions(system)
                    # structure, disassembler = self.simpleMoleculeUtility.structureType.assemble(**system)
                    # if self.bonds.isConnected(structure):
                    offsprings += (system,)
                system = {'molecules': molecules2, 'cell': cell}
                self.environmentUtility.putEnvironment(system, environment)
                atomSymbols, atomDistances, disassembler2 = self.simpleMoleculeUtility.getMinDistances(**system)
                minDistMatrix = self.ionDistances.getDistances(atomSymbols, self.conditions.externalPressure)
                if disassembler2.environment is not None:
                    inds = disassembler2.envIndices
                    atomDistances[tuple(np.meshgrid(inds, inds))] = minDistMatrix[tuple(np.meshgrid(inds, inds))]
                if np.all(atomDistances >= minDistMatrix):
                    self.conditions.putConditions(system)
                    # structure, disassembler = self.simpleMoleculeUtility.structureType.assemble(**system)
                    # if self.bonds.isConnected(structure):
                    offsprings += (system,)
                if offsprings:
                    return offsprings

        raise RuntimeError("Softmutation failed.")
