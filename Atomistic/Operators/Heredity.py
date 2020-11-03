import numpy as np


class Heredity:

    def __init__(self, Slab, cellProcessing):
        self.Slab = Slab
        self.cellProcessing = cellProcessing
        self.correlation = 0

    def __call__(self, system1, system2, *args, **kwargs):
        cell1 = system1['cell']
        molecules1 = system1['molecules']
        order1 = system1['order']
        cell2 = system2['cell']
        molecules2 = system1['molecules']
        order2 = system2['order']

        axis = np.random.randint(3)
        gaugesOfSlabs = tuple(np.random.randint(10,size=2).tolist())

        L = cell1.getAltitudes()[axis]
        Lchar = 0.5 * (cell1.getVolume() / len(order1)) ** (1 / 3) # average 'radius' of a molecule in the cell
        N = int(round(L / (Lchar + (L - Lchar) * (np.cos(self.correlation * np.pi / 2)) ** 2)))
        slabs1 = [self.Slab.getSlabs(molecules = molecules1, inputCell = cell1, outputCell = cell1,
                                     axis = axis, gaugesOfSlabs = gaugesOfSlabs,
                                     transformation = randomTransformation)
                  for randomTransformation in cell1.randomTransformations(N)]
        slabs1SortedByOrder = np.argsort(order1[slab1.indices].sum() for slab1, slab2 in slabs1)
        slab11, slab12 = slabs1[slabs1SortedByOrder[0]] if self.correlation > 0 else slabs1[slabs1SortedByOrder[-1]]



        slab21, slab22 = self.Slab.getSlabs(molecules = molecules2, inputCell = cell2, outputCell = cell2,
                                            axis = axis, gaugesOfSlabs = gaugesOfSlabs,
                                            origin = cell2.randomOrigin(), orientation = cell2.randomOrientation())

        outputCell = self.cellProcessing.getHybridCell(cell1, cell2)
