import unittest

from pathlib import Path

from ..RadialDistributionUtility import RadialDistributionUtility
from ...components import AtomisticRepresentation, AtomisticPoolEntry

PATH_WITH_TESTS = Path(__file__).parent


class RadialDistributionUtility_Test(unittest.TestCase):
    def setUp(self):
        self.utility = RadialDistributionUtility(symbols=['Mg', 'Al', 'O'])
        self.extension = self.utility.expressionExtension(self.utility)
        self.systemRDU1 = AtomisticPoolEntry(**AtomisticRepresentation.readAtomicStructure(PATH_WITH_TESTS/"systemRDU1.POSCAR"))
        self.systemRDU2 = AtomisticPoolEntry(**AtomisticRepresentation.readAtomicStructure(PATH_WITH_TESTS/"systemRDU2.POSCAR"))
        self.systemRDU3 = AtomisticPoolEntry(**AtomisticRepresentation.readAtomicStructure(PATH_WITH_TESTS/"systemRDU3.POSCAR"))

    def test_structureOrder(self):
        self.assertAlmostEqual(self.extension.structureOrder(self.systemRDU1), 0.207, places=3)
        self.assertAlmostEqual(self.extension.structureOrder(self.systemRDU2), 0.207, places=3)
        self.assertAlmostEqual(self.extension.structureOrder(self.systemRDU3), 0.169, places=3)

    def test_averageOrder(self):
        self.assertAlmostEqual(self.extension.averageOrder(self.systemRDU1), 0.22, places=2)
        self.assertAlmostEqual(self.extension.averageOrder(self.systemRDU2), 0.22, places=2)
        self.assertAlmostEqual(self.extension.averageOrder(self.systemRDU3), 0.21, places=2)

    def test_quasientropy(self):
        self.assertAlmostEqual(self.extension.quasientropy(self.systemRDU1), 0.072, places=3)
        self.assertAlmostEqual(self.extension.quasientropy(self.systemRDU2), 0.072, places=3)
        self.assertAlmostEqual(self.extension.quasientropy(self.systemRDU3), 0.166, places=3)

    def test_distance(self):
        self.systemRDU1.setProperty('radialDistributionUtility.complexFingerprint',
                                    self.extension.complexFingerprint(self.systemRDU1))
        self.systemRDU2.setProperty('radialDistributionUtility.complexFingerprint',
                                    self.extension.complexFingerprint(self.systemRDU2))
        self.systemRDU3.setProperty('radialDistributionUtility.complexFingerprint',
                                    self.extension.complexFingerprint(self.systemRDU3))
        self.assertAlmostEqual(self.utility.dist(self.systemRDU1, self.systemRDU2), 0, places=3)
        self.assertAlmostEqual(self.utility.dist(self.systemRDU2, self.systemRDU3), 0.6124, places=3)
        self.assertAlmostEqual(self.utility.dist(self.systemRDU1, self.systemRDU3), 0.6124, places=3)
