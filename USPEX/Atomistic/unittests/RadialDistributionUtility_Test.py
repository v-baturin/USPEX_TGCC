import os
import unittest
from os.path import join as pj

from ..RadialDistributionUtility import RadialDistributionUtility
from ...components import AtomisticRepresentation

PATH_WITH_TESTS = os.path.dirname(os.path.abspath(__file__))


class RadialDistributionUtility_Test(unittest.TestCase):
    def setUp(self):
        self.utility = RadialDistributionUtility(symbols=['Mg', 'Al', 'O'])
        self.systemRDU1 = AtomisticRepresentation.readAtomicStructure(pj(PATH_WITH_TESTS, "systemRDU1.POSCAR"))
        self.systemRDU2 = AtomisticRepresentation.readAtomicStructure(pj(PATH_WITH_TESTS, "systemRDU2.POSCAR"))
        self.systemRDU3 = AtomisticRepresentation.readAtomicStructure(pj(PATH_WITH_TESTS, "systemRDU3.POSCAR"))

    def test_structureOrder(self):
        self.assertAlmostEqual(self.utility.structureOrder(self.systemRDU1), 0.207, places=3)
        self.assertAlmostEqual(self.utility.structureOrder(self.systemRDU2), 0.207, places=3)
        self.assertAlmostEqual(self.utility.structureOrder(self.systemRDU3), 0.169, places=3)

    def test_averageOrder(self):
        self.assertAlmostEqual(self.utility.averageOrder(self.systemRDU1), 0.22, places=2)
        self.assertAlmostEqual(self.utility.averageOrder(self.systemRDU2), 0.22, places=2)
        self.assertAlmostEqual(self.utility.averageOrder(self.systemRDU3), 0.21, places=2)

    def test_quasientropy(self):
        self.assertAlmostEqual(self.utility.quasientropy(self.systemRDU1), 0.072, places=3)
        self.assertAlmostEqual(self.utility.quasientropy(self.systemRDU2), 0.072, places=3)
        self.assertAlmostEqual(self.utility.quasientropy(self.systemRDU3), 0.166, places=3)

    def test_distance(self):
        self.assertAlmostEqual(self.utility.dist(self.systemRDU1, self.systemRDU2), 0, places=3)
        self.assertAlmostEqual(self.utility.dist(self.systemRDU2, self.systemRDU3), 5.78101, places=3)
        self.assertAlmostEqual(self.utility.dist(self.systemRDU1, self.systemRDU3), 5.78101, places=3)
