import os
import unittest
from os.path import join as pj

from ..RadialDistributionUtility import RadialDistributionUtility
from ...components import AtomisticRepresentation

PATH_WITH_TESTS = os.path.dirname(os.path.abspath(__file__))


class RadialDistributionUtility_Test(unittest.TestCase):
    def setUp(self):
        self.utility = RadialDistributionUtility()
        with open(pj(PATH_WITH_TESTS, "systemRDU1.POSCAR"), 'rt') as f:
            self.systemRDU1 = AtomisticRepresentation.readAtomicStructure(f)
        with open(pj(PATH_WITH_TESTS, "systemRDU2.POSCAR"), 'rt') as f:
            self.systemRDU2 = AtomisticRepresentation.readAtomicStructure(f)
        with open(pj(PATH_WITH_TESTS, "systemRDU3.POSCAR"), 'rt') as f:
            self.systemRDU3 = AtomisticRepresentation.readAtomicStructure(f)

    def test_structureOrder(self):
        self.assertAlmostEqual(self.utility.structureOrder(self.systemRDU1), 2.531, places=3)
        self.assertAlmostEqual(self.utility.structureOrder(self.systemRDU2), 2.531, places=3)
        self.assertAlmostEqual(self.utility.structureOrder(self.systemRDU3), 2.059, places=3)

    def test_averageOrder(self):
        self.assertAlmostEqual(self.utility.averageOrder(self.systemRDU1), 2.87, places=2)
        self.assertAlmostEqual(self.utility.averageOrder(self.systemRDU2), 2.87, places=2)
        self.assertAlmostEqual(self.utility.averageOrder(self.systemRDU3), 2.68, places=2)

    def test_quasientropy(self):
        self.assertAlmostEqual(self.utility.quasientropy(self.systemRDU1), 0.092, places=3)
        self.assertAlmostEqual(self.utility.quasientropy(self.systemRDU2), 0.092, places=3)
        self.assertAlmostEqual(self.utility.quasientropy(self.systemRDU3), 0.190, places=3)

    def test_distance(self):
        self.assertTrue(self.utility.equal(self.systemRDU1, self.systemRDU2))
        self.assertFalse(self.utility.equal(self.systemRDU2, self.systemRDU3))
        self.assertFalse(self.utility.equal(self.systemRDU1, self.systemRDU3))
