"""
USPEX.Common.Atomistic.unittests.AtomicStructure_Test
=====================================================

Class for AtomicStricture testing

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import unittest
import os
import json
from ase.io import read


from ..Crystal import Crystal
from ..CrystalConfig import CrystalConfig

PATH_WITH_TESTS = os.path.dirname(os.path.abspath(__file__))


class AtomicStructure_Test(unittest.TestCase):

    def test_isMoleculesDistinct1(self):
        with open("{}/CNHO_1_system".format(PATH_WITH_TESTS), "rt") as f:
            system = Crystal.fromJSON(f.read())
        self.assertFalse(system.isMoleculesDistinct())

    def test_isMoleculesDistinct2(self):
        with open("{}/CNHO_2_system".format(PATH_WITH_TESTS), "rt") as f:
            system = Crystal.fromJSON(f.read())
        self.assertTrue(system.isMoleculesDistinct())

    def test_isGoodDistances1(self):
        with open("{}/CNHO_3_system".format(PATH_WITH_TESTS), "rt") as f:
            system = Crystal.fromJSON(f.read())
        self.assertFalse(system.isGoodDistances())

    def test_molecularFromJSON(self):
        with open(f'{PATH_WITH_TESTS}/h2o_nh3_2', 'rt') as f:
            string = f.read()
        self.assertRaises(AssertionError, Crystal.fromJSON, string)

    def test_isBad(self):
        with open("{}/CNHO_2_system".format(PATH_WITH_TESTS), "rt") as f:
            system = Crystal.fromJSON(f.read())
        self.assertFalse(system.isBad)
        system.markBad()
        self.assertTrue(system.isBad)

    def test_fingerprint(self):
        system1 = read(PATH_WITH_TESTS + '/system1_POSCAR')
        system1 = Crystal(symbols = system1.get_chemical_symbols(),
                                  positions = system1.get_positions(),
                                  cell = system1.get_cell())
        system2 = read(PATH_WITH_TESTS + '/system2_POSCAR')
        system2 = Crystal(symbols = system2.get_chemical_symbols(),
                                  positions = system2.get_positions(),
                                  cell = system2.get_cell())
        system1.fingerprintTolerance = 1.0e-6
        self.assertTrue(system1 == system2)
