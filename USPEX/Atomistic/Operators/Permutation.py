import numpy as np
from itertools import combinations
from copy import copy

from ..Transformation import Transformation
from ...DataModel.Entry import Entry
from ...DataModel.Flavour import FlavourFactory


_SWAP_ATTEMPTS = 1000


class Permutation:

    def __init__(self, utilities, suffix, howManySwaps = 5, specificSwaps = None, swapAttempts = _SWAP_ATTEMPTS):
        self.compositionSpace = utilities.compositionSpace
        self.environmentUtility = utilities.environmentUtility
        self.simpleMoleculeUtility = utilities.simpleMoleculeUtility
        self.bondUtility = utilities.bondUtility
        self.conditions = utilities.conditions
        self.cellUtility = utilities.cellUtility
        if len(self.compositionSpace.symbols) == 1:
            raise RuntimeError("Permutation does not work when number of symbols in calculation is 1.")
        self.specificSwaps = [self.compositionSpace.symbols[i-1] for i in specificSwaps] if specificSwaps is not None \
            else copy(self.compositionSpace.symbols)
        self.suffix = suffix
        self.howManySwaps = howManySwaps
        self.swapAttempts = swapAttempts

    def __call__(self, system: Entry, offspringFactory: FlavourFactory = None):
        molecules = system.getProperty('molecules', extension='atomistic', suffix=self.suffix)
        cell = system.getProperty('cell', extension='atomistic', suffix=self.suffix)
        structure = system.getProperty('structure', extension='atomistic', suffix=self.suffix)
        if self.cellUtility.isGoodCell(cell.getEnvelopeCell(structure.getCartesianCoordinates())):
            symbols = system['simpleMoleculeUtility.moleculeTypes.origin']

            swaps = [{i1,i2} for i1,i2 in combinations(range(len(molecules)), 2) if symbols[i1] != symbols[i2]
                     and symbols[i1] in self.specificSwaps and symbols[i2] in self.specificSwaps]

            for _ in range(self.swapAttempts):
                numberOfSwaps = np.random.randint(1, self.howManySwaps + 1)
                permutation = np.random.choice(swaps, numberOfSwaps)
                if len(set.union(*permutation)) == numberOfSwaps*2: # ensure that each molecule swapped not more than once
                    offspringMolecules = copy(molecules)

                    for i1, i2 in permutation:
                        transVec = molecules[i2].getCenterOfMassCartesianCoordinates() -\
                                           molecules[i1].getCenterOfMassCartesianCoordinates()
                        transformation = Transformation.fromRotVector([0.,0.,0.], transVec)
                        offspringMolecules[i1] = transformation.transform(molecules[i1])
                        offspringMolecules[i2] = (-transformation).transform(molecules[i2])

                    offspring = {'atomistic.molecules': offspringMolecules, 'atomistic.cell': cell}

                    offspring = offspringFactory(**offspring)
                    try:
                        offspring.setProperty('environments',
                                              system.getProperty('environments', extension='atomistic', suffix=self.suffix),
                                              extension='atomistic')
                    except Exception:
                        pass
                    structure = offspring.getProperty('structure', extension='atomistic')
                    minDistMatrix = self.bondUtility.getDistances(structure.getAtomTypes(),
                                                                  self.conditions.externalPressure)
                    if self.simpleMoleculeUtility.checkMinDistances(offspring, minDistMatrix):
                        self.conditions.putConditions(offspring)
                        # if self.bonds.isConnected(structure):
                        return (offspring,)

        raise RuntimeError("Permutation failed.")
