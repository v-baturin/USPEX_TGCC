import unittest

from pathlib import Path
import numpy as np

from ..RadialDistributionUtility import RadialDistributionUtility
from ...DataModel.Engine import Engine
from ...DataModel.Flavour import Flavour
from ...DataModel.Entry import Entry
from ...components import Atomistic

PATH_WITH_TESTS = Path(__file__).parent


class RadialDistributionUtility_Test(unittest.TestCase):
    def setUp(self):
        Engine.createEngine(':memory:')
        self.utility = RadialDistributionUtility(symbols=['Mg', 'Al', 'O'], suffix='origin')
        atomistic = Atomistic()
        extensions = dict(
            atomistic = atomistic.propertyExtension(),
            radialDistributionUtility=self.utility.propertyExtension()
        )

        self.systemRDU1 = Entry.newEntry(Flavour(extensions=extensions,
                                                          **{'.howCome': 'Seeds', '.parent': None},
                                                          **Atomistic.readAtomicStructure(PATH_WITH_TESTS/"systemRDU1.POSCAR")))
        self.systemRDU1.getProperty('structure', extension='atomistic')
        self.systemRDU2 = Entry.newEntry(Flavour(extensions=extensions,
                                                          **{'.howCome': 'Seeds', '.parent': None},
                                                           **Atomistic.readAtomicStructure(PATH_WITH_TESTS/"systemRDU2.POSCAR")))
        self.systemRDU2.getProperty('structure', extension='atomistic')
        self.systemRDU3 = Entry.newEntry(Flavour(extensions=extensions,
                                                          **{'.howCome': 'Seeds', '.parent': None},
                                                           **Atomistic.readAtomicStructure(PATH_WITH_TESTS/"systemRDU3.POSCAR")))
        self.systemRDU3.getProperty('structure', extension='atomistic')

    def test_structureOrder(self):
        self.assertAlmostEqual(self.systemRDU1['radialDistributionUtility.structureOrder.origin'], 0.017, places=3)
        self.assertAlmostEqual(self.systemRDU2['radialDistributionUtility.structureOrder.origin'], 0.017, places=3)
        self.assertAlmostEqual(self.systemRDU3['radialDistributionUtility.structureOrder.origin'], 0.014, places=3)

    def test_averageOrder(self):
        self.assertAlmostEqual(self.systemRDU1['radialDistributionUtility.averageOrder.origin'], 0.27, places=2)
        self.assertAlmostEqual(self.systemRDU2['radialDistributionUtility.averageOrder.origin'], 0.27, places=2)
        self.assertAlmostEqual(self.systemRDU3['radialDistributionUtility.averageOrder.origin'], 0.26, places=2)

    def test_quasientropy(self):
        self.assertAlmostEqual(self.systemRDU1['radialDistributionUtility.quasientropy.origin'], 0.052, places=3)
        self.assertAlmostEqual(self.systemRDU2['radialDistributionUtility.quasientropy.origin'], 0.052, places=3)
        self.assertAlmostEqual(self.systemRDU3['radialDistributionUtility.quasientropy.origin'], 0.111, places=3)

    def test_distance(self):
        self.assertAlmostEqual(self.utility.dist(self.systemRDU1, self.systemRDU2), 0, places=3)
        self.assertAlmostEqual(self.utility.dist(self.systemRDU2, self.systemRDU3), 0.4032, places=3)
        self.assertAlmostEqual(self.utility.dist(self.systemRDU1, self.systemRDU3), 0.4031, places=3)

    def test_make_matrices(self):
        from ..RadialDistributionUtility import _make_matrices
        input_params = {'coor': np.array([[0.87563798, 0.75493971, 0.30912173],
                                          [-0.74708733, 0.72292768, 0.47181632],
                                          [0.5909196, 0.89162408, 0.67711987],
                                          [-0.2956901, 0.4990522, 0.11302673],
                                          [-0.51877923, 0.59443453, 0.86884293],
                                          [-0.41634823, 0.84875506, 0.22431911],
                                          [-0.30433022, 0.19624119, 0.41174116],
                                          [-0.87838745, 0.12786159, 0.72795468],
                                          [-0.66942915, 0.78474586, 0.88697327],
                                          [-0.03048444, 0.10837592, 0.54401483],
                                          [-0.1128938, 0.50416208, 0.66472451],
                                          [-0.76648487, 0.2802966, 0.41314862]]),
                        'molIndices': [np.array([i]) for i in range(12)],
                        'envIndices': np.array([], dtype=int),
                        'lat': np.array([[12.346, 0., 0.],
                                         [0., 0.03160567, -4.61346253],
                                         [-0., -4.42362834, -0.03030516]]), 'numIons': np.array([12]), 'pbc': (1, 0, 0),
                        'Rmax': 10.0}
        print(input_params)
        _make_matrices(**input_params)
