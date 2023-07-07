import logging
logger = logging.getLogger(__name__)


import numpy as np
from collections import Counter

from ..Slab import Slab

ATTEMPTS = 100
NSLUBS = 2


class Heredity:

    def __init__(self, utilities, suffix='4', nslabs = None, attempts = ATTEMPTS, debug = False):
        self.cellUtility = utilities.cellUtility
        self.environmentUtility = utilities.environmentUtility
        self.compositionSpace = utilities.compositionSpace
        self.radialDistributionUtility = utilities.radialDistributionUtility
        self.simpleMoleculeUtility = utilities.simpleMoleculeUtility
        self.bondUtility = utilities.bondUtility
        self.conditions = utilities.conditions
        self.suffix = suffix
        self.nslabs = nslabs
        self.attempts = attempts
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        self.correlation = 0

    def tune(self, population, optType):
        fitness = [s[optType] for s in population]
        order = [system['radialDistributionUtility.averageOrder.origin'] for system in population]
        self.correlation = np.corrcoef(order, fitness)[0, 1]
        if np.isnan(self.correlation):
            self.correlation = 0

    def __call__(self, system1, system2, offspringFactory=None):
        molecules1 = system1.getProperty('molecules', prefix='atomistic', suffix=self.suffix)
        cell1 = system1.getProperty('cell', prefix='atomistic', suffix=self.suffix)
        composition1 = system1.getProperty('composition', prefix='simpleMoleculeUtility', suffix='origin')
        order1 = system1.getProperty('order', prefix='radialDistributionUtility', suffix=self.suffix)
        molecules2 = system2.getProperty('molecules', prefix='atomistic', suffix=self.suffix)
        cell2 = system2.getProperty('cell', prefix='atomistic', suffix=self.suffix)
        composition2 = system2.getProperty('composition', prefix='simpleMoleculeUtility', suffix='origin')
        order2 = system2.getProperty('order', prefix='radialDistributionUtility', suffix=self.suffix)
        try:
            system = np.random.choice((system1, system2))
            outputCell = system.getProperty('cell', prefix='atomistic', suffix=self.suffix)
            parentEnv = system.getProperty('environments', prefix='atomistic', suffix=self.suffix)
        except Exception:
            parentEnv = None
            outputCell = None

        for i in range(self.attempts):
            if outputCell is None:
                outputCell = self.cellUtility.getHybridCell(cell1, cell2, fraction=np.random.rand()).getOptimizedCell()
            if self.cellUtility.isGoodCell(outputCell):
                axis = np.random.randint(3)
                if self.nslabs is None:
                    if (composition1 == composition2) or outputCell.dim < 3:
                        nslabs = 2
                    else:
                        elementalComposition1 = self.simpleMoleculeUtility.getElementalComposition(composition1)
                        elementalComposition2 = self.simpleMoleculeUtility.getElementalComposition(composition2)
                        elements = set(elementalComposition1.keys()).union(set(elementalComposition2.keys()))
                        radii = np.fromiter((2 * el.covalent_radius for el in elements), dtype=float)
                        minSlice = radii.min()
                        maxSlice = radii.max()
                        medSlice = (minSlice + maxSlice) / 2
                        nslabs = int(np.round(outputCell.getCellParameters()[axis] / medSlice))
                        if nslabs < 2:
                            nslabs = 2
                else:
                    nslabs = self.nslabs

                gaugesOfSlabs = tuple(np.random.randint(3, 9, size=nslabs).tolist())

                if outputCell.dim == 0:
                    logger.debug(f"trying  {gaugesOfSlabs}-size slabs.")
                else:
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
                    offspring = {'atomistic.molecules': molecules, 'atomistic.cell': outputCell}
                    offspring = offspringFactory(**offspring)
                    if parentEnv is not None:
                        offspring.setProperty('environments', parentEnv, prefix='atomistic')
                    structure = offspring.getProperty('structure', prefix='atomistic')
                    minDistMatrix = self.bondUtility.getDistances(structure.getAtomTypes(),
                                                                  self.conditions.externalPressure)
                    if self.simpleMoleculeUtility.checkMinDistances(offspring, minDistMatrix):
                        self.conditions.putConditions(offspring)
                        # if self.bonds.isConnected(structure):
                        return offspring,


        raise RuntimeError("Heredity failed.")

