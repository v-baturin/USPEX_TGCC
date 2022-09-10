import numpy as np
from itertools import combinations
from copy import copy

from ..Transformation import Transformation


_SWAP_ATTEMPTS = 1000


class Permutation:

    def __init__(self, utilities, howManySwaps = 5, specificSwaps = None, swapAttempts = _SWAP_ATTEMPTS):
        self.compositionSpace = utilities.compositionSpace
        self.environmentUtility = utilities.environmentUtility
        self.simpleMoleculeUtility = utilities.simpleMoleculeUtility
        self.ionDistances = utilities.ionDistances
        self.bonds = utilities.bonds
        self.conditions = utilities.conditions
        self.cellUtility = utilities.cellUtility
        if len(self.compositionSpace.symbols) == 1:
            raise RuntimeError("Permutation does not work when number of symbols in calculation is 1.")
        self.specificSwaps = [self.compositionSpace.symbols[i-1] for i in specificSwaps] if specificSwaps is not None \
            else copy(self.compositionSpace.symbols)
        self.howManySwaps = howManySwaps
        self.swapAttempts = swapAttempts

    def __call__(self, system, *args, **kwargs):
        molecules = system['molecules']
        cell = system['cell']
        environment = system['environment'] if 'environment' in system else None
        structure, disassembler = self.simpleMoleculeUtility.structureType.assemble(molecules, cell)
        if self.cellUtility.isGoodCell(cell.getEnvelopeCell(structure.getCartesianCoordinates())):
            symbols = self.simpleMoleculeUtility.moleculeTypes(system)

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

                    offspring = {'molecules': offspringMolecules, 'cell': cell}
                    self.environmentUtility.putEnvironment(offspring, environment)
                    atomSymbols, atomDistances, disassembler = self.simpleMoleculeUtility.getMinDistances(**offspring)
                    minDistMatrix = self.ionDistances.getDistances(atomSymbols, self.conditions.externalPressure)
                    if disassembler.environment is not None:
                        inds = disassembler.envIndices
                        atomDistances[tuple(np.meshgrid(inds, inds))] = minDistMatrix[tuple(np.meshgrid(inds, inds))]
                    if np.all(atomDistances >= minDistMatrix):
                        self.conditions.putConditions(offspring)
                        # structure, disassembler = self.simpleMoleculeUtility.structureType.assemble(**offspring)
                        # if self.bonds.isConnected(structure):
                        return (offspring,)

        raise RuntimeError("Permutation failed.")
