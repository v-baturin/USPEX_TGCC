import numpy as np

from ..Slab import Slab


class Heredity:

    def __init__(self, cellUtility, compositionSpace, radialDistributionUtility):
        self.cellUtility = cellUtility
        self.compositionSpace = compositionSpace
        self.radialDistributionUtility = radialDistributionUtility
        self.correlation = 0

    def __call__(self, system1, system2, *args, **kwargs):
        cell1 = system1['cell']
        molecules1 = system1['molecules']
        order1 = self.radialDistributionUtility.order(system1)
        cell2 = system2['cell']
        molecules2 = system1['molecules']
        order2 = self.radialDistributionUtility.order(system2)

        axis = np.random.randint(3)
        gaugesOfSlabs = tuple(np.random.randint(10, size=2).tolist())

        slab11, slab12 = Slab.getRandomSlabs(molecules=molecules1, inputCell=cell1, outputCell=cell1,
                                             axis=axis, gaugesOfSlabs=gaugesOfSlabs,
                                             order=order1, correlation=self.correlation, parity=0)

        slab21, slab22 = Slab.getRandomSlabs(molecules=molecules2, inputCell=cell2, outputCell=cell2,
                                             axis=axis, gaugesOfSlabs=gaugesOfSlabs,
                                             order=order2, correlation=self.correlation, parity=1)

        goodCandidateMolecules = molecules1[slab11.indices] + molecules2[slab22.indices]
        goodCandidateOrder = slab11.depths + slab22.depths
        goodCandidateMolecules = goodCandidateMolecules[np.argsort(goodCandidateOrder)]

        badCandidateMolecules = molecules2[slab21.indices] + molecules1[slab12.indices]
        badCandidateOrder = slab21.depths + slab12.depths
        badCandidateMolecules = badCandidateMolecules[np.argsort(badCandidateOrder)]

        outputCell = self.cellUtility.getHybridCell(cell1, cell2)

        composition1 = self.compositionSpace.calculateComposition(molecules1)
        composition2 = self.compositionSpace.calculateComposition(molecules2)
        composition = self.compositionSpace.calculateComposition(goodCandidateMolecules)
        desiredComposition = self.compositionSpace.findDesiredComposition(composition1, composition2, composition)[0]

        goodCandidateMolecules = removeExtra(goodCandidateMolecules, self.compositionSpace, desiredComposition)
        goodCandidateMolecules += addLacking(badCandidateMolecules,
                                             desiredComposition - self.compositionSpace.calculateComposition(goodCandidateMolecules))

        return ({'molecules' : goodCandidateMolecules, 'cell': outputCell},)


def removeExtra(molecules, compositionSpace, desiredComposition):
    """
    remove extra molecules from child_good structure
    :param molecules:
    :param compositionSpace:
    :param desiredComposition:
    :return:
    """
    moleculesNew = []
    for molecule in molecules:
        if compositionSpace.calculateComposition(moleculesNew)[molecule] < desiredComposition[molecule]:
            moleculesNew.append(molecule)
    return moleculesNew


def addLacking(molecules, desiredComposition):
    """
    add lacking molecules to good_child from bad_child
    :param molecules:
    :param desiredComposition:
    :return:
    """
    moleculesNew = []
    for symbol, desired in desiredComposition.items():
        moleculesIter = iter(molecules)
        while desired > 0:
            molecule = next(moleculesIter)
            if molecule.symbol == symbol:
                moleculesNew.append(molecule)
                desired -= 1
    return moleculesNew
