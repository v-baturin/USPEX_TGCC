import numpy as np
from itertools import combinations
from copy import copy

from ..Transformation import Transformation


_SWAP_ATTEMPTS = 1000


class Permutation:

    def __init__(self, utilities, howManySwaps = 5, swapAttempts = _SWAP_ATTEMPTS):
        self.compositionSpace = utilities.compositionSpace
        self.simpleMoleculeUtility = utilities.simpleMoleculeUtility
        self.ionDistances = utilities.ionDistances
        self.conditions = utilities.conditions
        if len(self.compositionSpace.symbols) == 1:
            raise RuntimeError("Permutation does not work when number of symbols in calculation is 1.")
        self.specificSwaps = []
        self.howManySwaps = howManySwaps
        self.swapAttempts = swapAttempts

    def __call__(self, system, *args, **kwargs):
        molecules = system['molecules']
        cell = system['cell']
        symbols = self.simpleMoleculeUtility.moleculeTypes(system)

        swaps = [{i1,i2} for i1,i2 in combinations(range(len(molecules)), 2) if symbols[i1] != symbols[i2]]

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

                atomSymbols, atomDistances = self.simpleMoleculeUtility.getMinDistances(offspringMolecules, cell)
                minDistMatrix = self.ionDistances.getDistances(atomSymbols, self.conditions)
                if np.all(atomDistances >= minDistMatrix):
                    offspring = {'molecules': offspringMolecules, 'cell': cell}
                    self.conditions.putConditions(offspring)
                    return (offspring,)

        raise RuntimeError("Permutation failed.")
