import os
import unittest
from os.path import join as pj
from ase.io.vasp import read_vasp

from ..RadialDistributionUtility import RadialDistributionUtility
from ..Crystal import Crystal

PATH_WITH_TESTS = os.path.dirname(os.path.abspath(__file__))


class RadialDistributionUtility_Test(unittest.TestCase):
    def setUp(self):
        self.utility = RadialDistributionUtility()
        self.systemRDU1 = {}
        self.systemRDU2 = {}
        self.systemRDU3 = {}

        tmp1 = read_vasp(pj(PATH_WITH_TESTS, "systemRDU1.POSCAR"))
        self.systemRDU1['structure'] = Crystal(symbols=tmp1.get_chemical_symbols(),
                                               cell=tmp1.get_cell(),
                                               positions=tmp1.get_positions())

        tmp2 = read_vasp(pj(PATH_WITH_TESTS, "systemRDU2.POSCAR"))
        self.systemRDU2['structure'] = Crystal(symbols=tmp2.get_chemical_symbols(),
                                               cell=tmp2.get_cell(),
                                               positions=tmp2.get_positions())

        tmp3 = read_vasp(pj(PATH_WITH_TESTS, "systemRDU3.POSCAR"))
        self.systemRDU3['structure'] = Crystal(symbols=tmp3.get_chemical_symbols(),
                                               cell=tmp3.get_cell(),
                                               positions=tmp3.get_positions())


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
