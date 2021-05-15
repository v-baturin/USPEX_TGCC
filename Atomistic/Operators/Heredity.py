import logging
logger = logging.getLogger(__name__)


import numpy as np
from collections import Counter

from ..Slab import Slab

ATTEMPTS = 100
NSLUBS = 2


class Heredity:

    def __init__(self, utilities, nslubs = None, attempts = ATTEMPTS, debug = False):
        self.cellUtility = utilities.cellUtility
        self.compositionSpace = utilities.compositionSpace
        self.radialDistributionUtility = utilities.radialDistributionUtility
        self.simpleMoleculeUtility = utilities.simpleMoleculeUtility
        self.ionDistances = utilities.ionDistances
        self.conditions = utilities.conditions
        self.nslubs = nslubs
        self.attempts = attempts
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        self.correlation = 0

    def tune(self, population, allFitnesses):
        fitness = [allFitnesses[s['ID']] for s in population]
        order = [self.radialDistributionUtility.averageOrder(system) for system in population]
        self.correlation = np.corrcoef(order, fitness)[0,1]
        if np.isnan(self.correlation):
            self.correlation = 0

    def __call__(self, system1, system2):
        cell1 = system1['cell']
        molecules1 = system1['molecules']
        composition1 = self.simpleMoleculeUtility.composition(system1)
        order1 = self.radialDistributionUtility.order(system1)
        cell2 = system2['cell']
        molecules2 = system2['molecules']
        composition2 = self.simpleMoleculeUtility.composition(system2)
        order2 = self.radialDistributionUtility.order(system2)

        for i in range(self.attempts):
            outputCell = self.cellUtility.getHybridCell(cell1, cell2, fraction=np.random.rand())
            axis = np.random.randint(3)
            if self.nslubs is None:
                if composition1 == composition2:
                    nslubs = 2
                else:
                    elementalComposition1 = self.simpleMoleculeUtility.getElementalComposition(composition1)
                    elementalComposition2 = self.simpleMoleculeUtility.getElementalComposition(composition2)
                    elements = set(elementalComposition1.keys()).union(set(elementalComposition2.keys()))
                    radii = np.fromiter((2 * el.covalent_radius for el in elements), dtype=float)
                    minSlice = radii.min()
                    maxSlice = radii.max()
                    medSlice = (minSlice + maxSlice) / 2
                    nslubs = int(np.round(outputCell.getCellParameters()[axis] / medSlice))
                    if nslubs < 2:
                        nslubs = 2
            else:
                nslubs = self.nslubs

            gaugesOfSlabs = tuple(np.random.randint(3, 9, size=nslubs).tolist())

            logger.debug(f"trying {outputCell.getCellParameters()} cell and {gaugesOfSlabs}-size slabs.")

            slabs1 = Slab.getRandomSlabs(molecules=molecules1, inputCell=cell1, outputCell=outputCell,
                                         axis=axis, gaugesOfSlabs=gaugesOfSlabs,
                                         order=order1, correlation=self.correlation, parity=0)

            slabs2 = Slab.getRandomSlabs(molecules=molecules2, inputCell=cell2, outputCell=outputCell,
                                         axis=axis, gaugesOfSlabs=gaugesOfSlabs,
                                         order=order2, correlation=self.correlation, parity=1)

            goodCandidateMolecules = []
            goodCandidateDepths = []
            badCandidateMolecules = []
            badCandidateDepths = []

            parity = 0
            for slab1, slab2 in zip(slabs1, slabs2):
                if parity == 0:
                    goodCandidateMolecules.extend(slab1.molecules)
                    goodCandidateDepths.extend(slab1.depths)
                    badCandidateMolecules.extend(slab2.molecules)
                    badCandidateDepths.extend(slab2.depths)
                    parity = 1
                else:
                    badCandidateMolecules.extend(slab1.molecules)
                    badCandidateDepths.extend(slab1.depths)
                    goodCandidateMolecules.extend(slab2.molecules)
                    goodCandidateDepths.extend(slab2.depths)
                    parity = 0

            goodCandidateMolecules = [goodCandidateMolecules[i] for i in reversed(np.argsort(goodCandidateDepths))]
            badCandidateMolecules = [badCandidateMolecules[i] for i in np.argsort(badCandidateDepths)]

            goodCandidateMoleculeTypes = [self.simpleMoleculeUtility.determineMoleculeType(molecule) for molecule in goodCandidateMolecules]
            badCandidateMoleculeTypes = [self.simpleMoleculeUtility.determineMoleculeType(molecule) for molecule in badCandidateMolecules]
            composition = Counter(dict(zip(*np.unique(goodCandidateMoleculeTypes, return_counts=True))))
            desiredComposition = self.compositionSpace.findDesiredComposition(composition1 + composition2, composition)
            goodCandidateIndices = self.compositionSpace.choose(goodCandidateMoleculeTypes, desiredComposition)
            badCandidateIndices = self.compositionSpace.choose(badCandidateMoleculeTypes, desiredComposition - composition)

            molecules = [goodCandidateMolecules[i] for i in goodCandidateIndices] + \
                        [badCandidateMolecules[i] for i in badCandidateIndices]
            moleculeTypes = [self.simpleMoleculeUtility.determineMoleculeType(molecule) for molecule in molecules]
            composition = Counter(dict(zip(*np.unique(moleculeTypes, return_counts=True))))
            if composition == desiredComposition:
                atomSymbols, atomDistances = self.simpleMoleculeUtility.getMinDistances(molecules, outputCell)
                minDistMatrix = self.ionDistances.getDistances(atomSymbols, self.conditions)
                if np.all(atomDistances >= minDistMatrix):
                    system = {'molecules': molecules, 'cell': outputCell}
                    self.conditions.putConditions(system)
                    return (system,)

        raise RuntimeError("Heredity failed.")

