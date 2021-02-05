import logging
logger = logging.getLogger(__name__)

import numpy as np

from ..Transformation import Transformation
from ...Atomistic.softmodes.calcSoftModes import calcSoftModes


_MIN_VALID_FREQUENCY = 5.0e-4


class Softmodemutation:
    def __init__(self, simpleMoleculeUtility, ionDistances, conditions, degree: float = None):
        self.simpleMoleculeUtility = simpleMoleculeUtility
        self.ionDistances = ionDistances
        self.conditions = conditions
        self.degree= degree
        self.knownSystems = {}

    def __call__(self, system):
        ID = system['ID']
        molecules = system['molecules']
        cell = system['cell']
        structure, disassembler = self.simpleMoleculeUtility.atomicStructureFactory.assemble(molecules, cell)
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
            degree = self.degree if self.degree else system.covalentRadii.mean() * 3
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
                molecules1.extend(molecule1)
                molecules2.extend(molecule2)

            offsprings = ()
            atomSymbols, atomDistances = self.simpleMoleculeUtility.getMinDistances(molecules1, cell)
            minDistMatrix = self.ionDistances.getDistances(atomSymbols, self.conditions)
            if atomDistances >= minDistMatrix:
                offsprings += ({'molecules' : molecules, 'cell': cell},)
            atomSymbols, atomDistances = self.simpleMoleculeUtility.getMinDistances(molecules2, cell)
            minDistMatrix = self.ionDistances.getDistances(atomSymbols, self.conditions)
            if atomDistances >= minDistMatrix:
                offsprings += ({'molecules' : molecules, 'cell': cell},)
            if offsprings:
                return offsprings

        raise RuntimeError("RandTop failed.")
