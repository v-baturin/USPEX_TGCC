import logging
logger = logging.getLogger(__name__)

import numpy as np

from ..Transformation import Transformation


_MIN_VALID_FREQUENCY = 5.0e-4


class Softmodemutation:
    def __init__(self, utilities, degree: float = None):
        self.simpleMoleculeUtility = utilities.simpleMoleculeUtility
        self.bondUtility = utilities.bondUtility
        self.environmentUtility = utilities.environmentUtility
        self.conditions = utilities.conditions
        self.cellUtility = utilities.cellUtility
        self.degree= degree
        self.knownSystems = {}

    def __call__(self, system, offspringFactory=None):
        ID = system['ID']
        molecules = system['molecules']
        cell = system['cell']
        structure, disassembler = system.atomicDisassemblerType.assemble(molecules, cell)
        if self.cellUtility.isGoodCell(cell.getEnvelopeCell(structure.getCartesianCoordinates())):
            degree = self.degree if self.degree else np.mean([el.covalent_radius for el in structure.getAtomTypes()]) * 3
            if ID in self.knownSystems:
                frequencies, eigenVectors = self.knownSystems[ID]
            else:
                try:
                    bonds = self.bondUtility.getMinimalGraphBonds(structure)
                except Exception as e:
                    logger.debug(e)
                    raise RuntimeError("Softmutation failed.")
                frequencies, eigenVectors = self.bondUtility.calcSoftModes(structure, bonds)
                self.knownSystems[ID] = (frequencies, eigenVectors)
            while len(frequencies) > 0:
                freq = frequencies.pop(0)
                eigenVector = eigenVectors.pop(0)
                if freq < _MIN_VALID_FREQUENCY:
                    continue
                displacements = eigenVector.reshape((len(structure), 3))
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
                offspring1 = {'molecules': molecules1, 'cell': cell}
                if 'environments' in system:
                    offspring1['environments'] = system['environments']
                offspring1 = offspringFactory(**offspring1)
                structure1 = offspring1.getAtomicStructure()
                minDistMatrix = self.bondUtility.getDistances(structure1.getAtomTypes(),
                                                              self.conditions.externalPressure)
                if self.simpleMoleculeUtility.checkMinDistances(offspring1, minDistMatrix):
                    self.conditions.putConditions(offspring1)
                    if self.bondUtility.isConnected(structure1):
                        offsprings += (offspring1,)
                offspring2 = {'molecules': molecules2, 'cell': cell}
                if 'environments' in system:
                    offspring2['environments'] = system['environments']
                offspring2 = offspringFactory(**offspring2)
                structure2 = offspring2.getAtomicStructure()
                minDistMatrix = self.bondUtility.getDistances(structure2.getAtomTypes(),
                                                              self.conditions.externalPressure)
                if self.simpleMoleculeUtility.checkMinDistances(offspring2, minDistMatrix):
                    self.conditions.putConditions(offspring2)
                    if self.bondUtility.isConnected(structure2):
                        offsprings += (offspring2,)
                if offsprings:
                    return offsprings

        raise RuntimeError("Softmutation failed.")
