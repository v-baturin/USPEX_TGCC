import logging
logger = logging.getLogger(__name__)

import numpy as np
from copy import copy

from ..Transformation import Transformation


class Rotation():

    HOW_MANY_ATTEMPTS_ROTATION = 100

    def __init__(self, simpleMoleculeUtility, ionDistances, conditions):
        self.simpleMoleculeUtility = simpleMoleculeUtility
        self.ionDistances = ionDistances
        self.conditions = conditions


    def __call__(self, system) -> tuple:
        molecules = system['molecules']
        cell = system['cell']

        for i in range(self.HOW_MANY_ATTEMPTS_ROTATION):
            totalNumMols = len(molecules)
            for j in np.random.choice(totalNumMols, np.random.randint(totalNumMols), replace=False):
                molecule = molecules[j]
                offset = Transformation.fromRotVector([0.,0.,0.,], molecule.getCenterOfMassCartesianCoordinates())
                molecule = (-offset).transform(molecule)
                inertiaValues, inertiaVectors = molecule.getPrincipleAxes()
                rotationClearance = self.simpleMoleculeUtility.rotationClearance(inertiaValues)
                rotVector = (inertiaVectors*rotationClearance)[np.random.randint(3)]*(2*np.random.random_sample() - 1)
                transformation = Transformation.fromRotVector(rotVector, offset.transVec)
                molecule = transformation.transform(molecule)
                molecules[j] = molecule
                # molecules[j] = self.simpleMoleculeUtility.rotateFlexDiherdal(molecule, dihedral, dihedralAngle)

            atomSymbols, atomDistances = self.simpleMoleculeUtility.getMinDistances(molecules, cell)
            minDistMatrix = self.ionDistances.getDistances(atomSymbols, self.conditions)
            if atomDistances >= minDistMatrix:
                return ({'molecules' : molecules, 'cell': cell},)

        raise RuntimeError("Rotation failed.")
