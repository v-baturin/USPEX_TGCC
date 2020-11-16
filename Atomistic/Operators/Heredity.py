import numpy as np


class Heredity:

    def __init__(self, Slab, cellProcessing, compositionSpace):
        self.Slab = Slab
        self.cellProcessing = cellProcessing
        self.compositionSpace = compositionSpace
        self.correlation = 0

    def __call__(self, system1, system2, *args, **kwargs):
        cell1 = system1['cell']
        molecules1 = system1['molecules']
        order1 = system1['order']
        cell2 = system2['cell']
        molecules2 = system1['molecules']
        order2 = system2['order']

        axis = np.random.randint(3)
        gaugesOfSlabs = tuple(np.random.randint(10, size=2).tolist())

        slab11, slab12 = self.Slab.getRandomSlabs(molecules=molecules1, inputCell=cell1, outputCell=cell1,
                                                  axis=axis, gaugesOfSlabs=gaugesOfSlabs,
                                                  order=order1, correlation=self.correlation, parity=0)

        slab21, slab22 = self.Slab.getRandomSlabs(molecules=molecules2, inputCell=cell2, outputCell=cell2,
                                                  axis=axis, gaugesOfSlabs=gaugesOfSlabs,
                                                  order=order2, correlation=self.correlation, parity=1)

        goodCandidateMolecules = molecules1[slab11.indices] + molecules2[slab22.indices]
        goodCandidateOrder = order1[slab11.indices] + order2[slab22.indices]

        badCandidateMolecules = molecules2[slab21.indices] + molecules1[slab12.indices]
        badCandidateOrder = order2[slab21.indices] + order1[slab12.indices]

        outputCell = self.cellProcessing.getHybridCell(cell1, cell2)

        composition1 = self.compositionSpace.calculateComposition(molecules1)
        composition2 = self.compositionSpace.calculateComposition(molecules2)
        composition = self.compositionSpace.calculateComposition(goodCandidateMolecules)
        desiredComposition = self.compositionSpace.findDesiredComposition(composition1, composition2, composition)[0]

        goodCandidateMolecules = removeExtra(goodCandidateMolecules[np.argsort(goodCandidateOrder)],
                                             self.compositionSpace, desiredComposition)

        composition = self.compositionSpace.calculateComposition(goodCandidateMolecules)
        goodCandidateMolecules += addLacking(badCandidateMolecules[np.argsort(badCandidateOrder)],
                                             desiredComposition - composition)


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
