import numpy as np
from collections import Counter

from ..Slab import Slab

ATTEMPTS = 10
NSLUBS = 2


class Heredity:

    def __init__(self, utilities, nslubs = NSLUBS, attempts = ATTEMPTS):
        self.cellUtility = utilities.cellUtility
        self.compositionSpace = utilities.compositionSpace
        self.radialDistributionUtility = utilities.radialDistributionUtility
        self.simpleMoleculeUtility = utilities.simpleMoleculeUtility
        self.ionDistances = utilities.ionDistances
        self.conditions = utilities.conditions
        self.nslubs = nslubs
        self.attempts = attempts
        self.correlation = 0

    def __call__(self, system1, system2):
        cell1 = system1['cell']
        molecules1 = system1['molecules']
        moleculeTypes1 = self.simpleMoleculeUtility.moleculeTypes(system1)
        composition1 = self.simpleMoleculeUtility.composition(system1)
        order1 = self.radialDistributionUtility.order(system1)
        cell2 = system2['cell']
        molecules2 = system1['molecules']
        moleculeTypes2 = self.simpleMoleculeUtility.moleculeTypes(system2)
        composition2 = self.simpleMoleculeUtility.composition(system2)
        order2 = self.radialDistributionUtility.order(system2)

        outputCell = self.cellUtility.getHybridCell(cell1, cell2, fraction = np.random.rand())

        for i in range(self.attempts):
            axis = np.random.randint(3)
            gaugesOfSlabs = tuple(np.random.randint(10, size=self.nslubs).tolist())

            slabs1 = Slab.getRandomSlabs(molecules=molecules1, inputCell=cell1, outputCell=outputCell,
                                         axis=axis, gaugesOfSlabs=gaugesOfSlabs,
                                         order=order1, correlation=self.correlation, parity=0)

            slabs2 = Slab.getRandomSlabs(molecules=molecules2, inputCell=cell2, outputCell=outputCell,
                                         axis=axis, gaugesOfSlabs=gaugesOfSlabs,
                                         order=order2, correlation=self.correlation, parity=1)

            goodCandidateMolecules = []
            goodCandidateMoleculeTypes = []
            goodCandidateDepths = []

            badCandidateMolecules = []
            badCandidateMoleculeTypes = []
            badCandidateDepths = []

            parity = 0
            for slab1, slab2 in zip(slabs1, slabs2):
                if not parity:
                    goodCandidateMolecules.extend(slab1.transformation.transform(molecules1[i]) for i in slab1.indices)
                    goodCandidateMoleculeTypes.extend(moleculeTypes1[i] for i in slab1.indices)
                    goodCandidateDepths.extend(slab1.depths)
                    badCandidateMolecules.extend(slab2.transformation.transform(molecules2[i]) for i in slab2.indices)
                    badCandidateMoleculeTypes.extend(moleculeTypes2[i] for i in slab2.indices)
                    badCandidateDepths.extend(slab2.depths)
                    parity = 1
                else:
                    badCandidateMolecules.extend(slab1.transformation.transform(molecules1[i]) for i in slab1.indices)
                    badCandidateMoleculeTypes.extend(moleculeTypes1[i] for i in slab1.indices)
                    badCandidateDepths.extend(slab1.depths)
                    goodCandidateMolecules.extend(slab2.transformation.transform(molecules2[i]) for i in slab2.indices)
                    goodCandidateMoleculeTypes.extend(moleculeTypes2[i] for i in slab2.indices)
                    goodCandidateDepths.extend(slab2.depths)
                    parity = 0


            goodCandidateSortOrder = reversed(np.argsort(goodCandidateDepths))
            goodCandidateMolecules = [goodCandidateMolecules[i] for i in goodCandidateSortOrder]
            goodCandidateMoleculeTypes = [goodCandidateMoleculeTypes[i] for i in goodCandidateSortOrder]
            badCandidateSortOrder = np.argsort(badCandidateDepths)
            badCandidateMolecules = [badCandidateMolecules[i] for i in badCandidateSortOrder]
            badCandidateMoleculeTypes = [badCandidateMoleculeTypes[i] for i in badCandidateSortOrder]

            composition = Counter(dict(zip(*np.unique(goodCandidateMoleculeTypes, return_counts=True))))
            desiredComposition = self.compositionSpace.findDesiredComposition(composition1 + composition2, composition)

            goodCandidateIndices = self.compositionSpace.choose(goodCandidateMoleculeTypes, desiredComposition)
            badCandidateIndices = self.compositionSpace.choose(badCandidateMoleculeTypes, desiredComposition - composition)

            molecules = [goodCandidateMolecules[i] for i in goodCandidateIndices] + \
                        [badCandidateMolecules[i] for i in badCandidateIndices]

            atomSymbols, atomDistances = self.simpleMoleculeUtility.getMinDistances(molecules, outputCell)
            minDistMatrix = self.ionDistances.getDistances(atomSymbols, self.conditions)
            if atomDistances >= minDistMatrix:
                return ({'molecules' : molecules, 'cell': outputCell},)

        raise RuntimeError("Heredity failed.")

