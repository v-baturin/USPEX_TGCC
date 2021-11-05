import logging
logger = logging.getLogger(__name__)

import numpy as np

from ..Transformation import Transformation
from ...Atomistic.softmodes.calcSoftModes import calcSoftModes


_MIN_VALID_FREQUENCY = 5.0e-4


class Softmodemutation:
    def __init__(self, utilities, degree: float = None):
        self.cellUtility = utilities.cellUtility
        self.simpleMoleculeUtility = utilities.simpleMoleculeUtility
        self.ionDistances = utilities.ionDistances
        self.world = utilities.world
        self.conditions = utilities.conditions
        self.degree= degree
        self.knownSystems = {}

    def __call__(self, system):
        ID = system['ID']
        molecules = system['molecules']
        cell = system['cell']
        structure, disassembler = self.simpleMoleculeUtility.structureType.assemble(molecules, cell)
        degree = self.degree if self.degree else np.mean([el.covalent_radius for el in structure.getAtomTypes()]) * 3
        if ID in self.knownSystems:
            frequencies, eigenVectors = self.knownSystems[ID]
        else:
            frequencies, eigenVectors = calcSoftModes(structure)
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

            # change the cell parameters to the user-given ones in case of fixed cell calculations
            try:
                cell = self.cellUtility.getCell()
            except RuntimeError:
                pass

            offsprings = ()
            atomSymbols, atomDistances = self.simpleMoleculeUtility.getMinDistances(molecules1, cell)
            minDistMatrix = self.ionDistances.getDistances(atomSymbols, self.conditions.externalPressure)
            if np.all(atomDistances >= minDistMatrix):
                system = {'molecules': molecules1, 'cell': cell}
                self.world.putEnvironment(system)
                self.conditions.putConditions(system)
                offsprings += (system,)
            atomSymbols, atomDistances = self.simpleMoleculeUtility.getMinDistances(molecules2, cell)
            minDistMatrix = self.ionDistances.getDistances(atomSymbols, self.conditions.externalPressure)
            if np.all(atomDistances >= minDistMatrix):
                system = {'molecules': molecules2, 'cell': cell}
                self.world.putEnvironment(system)
                self.conditions.putConditions(system)
                offsprings += (system,)
            if offsprings:
                return offsprings

        raise RuntimeError("Softmutation failed.")
