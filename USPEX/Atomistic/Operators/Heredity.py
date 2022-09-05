import logging
logger = logging.getLogger(__name__)


import numpy as np
from collections import Counter

from ..Slab import Slab

ATTEMPTS = 100
NSLUBS = 2


class Heredity:

    def __init__(self, utilities, nslabs = None, attempts = ATTEMPTS, debug = False):
        self.cellUtility = utilities.cellUtility
        self.environmentUtility = utilities.environmentUtility
        self.compositionSpace = utilities.compositionSpace
        self.radialDistributionUtility = utilities.radialDistributionUtility
        self.simpleMoleculeUtility = utilities.simpleMoleculeUtility
        self.ionDistances = utilities.ionDistances
        self.bonds = utilities.bonds
        self.conditions = utilities.conditions
        self.nslabs = nslabs
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
            if self.environmentUtility.hasEnvironment():
                environment = self.environmentUtility.getRandomEnvironment()
            else:
                environment = None
            self.cellUtility.communicateWithEnvironment(environment)
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
                    system = {'molecules': molecules, 'cell': outputCell}
                    self.environmentUtility.putEnvironment(system)
                    atomSymbols, atomDistances, disassembler = self.simpleMoleculeUtility.getMinDistances(**system)
                    minDistMatrix = self.ionDistances.getDistances(atomSymbols, self.conditions.externalPressure)
                    if disassembler.environment is not None:
                        inds = disassembler.envIndices
                        atomDistances[tuple(np.meshgrid(inds, inds))] = minDistMatrix[tuple(np.meshgrid(inds, inds))]
                    if np.all(atomDistances >= minDistMatrix):
                        self.conditions.putConditions(system)
                        # structure, disassembler = self.simpleMoleculeUtility.structureType.assemble(**system)
                        # if self.bonds.isConnected(structure):
                        return (system,)


        raise RuntimeError("Heredity failed.")

