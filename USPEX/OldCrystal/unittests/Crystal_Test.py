"""
USPEX.Common.Atomistic.unittests.AtomicStructure_Test
=====================================================

Class for AtomicStricture testing

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import unittest
import os
import toml
from ase.io import read
import numpy as np


from ..Crystal import Crystal

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

    def test_problem_1(self):
        tmp = read('{}/diamond8.vasp'.format(PATH_WITH_TESTS))
        self.diamond8 = Crystal(symbols=tmp.get_chemical_symbols(),
                                        scaled_positions=tmp.get_scaled_positions(),
                                        cell=tmp.get_cell())
        self.assertTrue(self.diamond8.isGoodSystem())

    def test_problem_2(self):
        tmp = read('{}/dia_2x2x2.vasp'.format(PATH_WITH_TESTS))
        self.dia_2x2x2 = Crystal(symbols=tmp.get_chemical_symbols(),
                                         scaled_positions=tmp.get_scaled_positions(),
                                         cell=tmp.get_cell())
        self.assertTrue(self.dia_2x2x2.isGoodSystem())

    def test_decomposeDisplacemants_atomic(self):
        with open(os.path.join(PATH_WITH_TESTS, 'system1.toml'), 'rt') as f:
            repr1 = toml.load(f)
        system1 = Crystal.fromDICT(repr1, old = False)
        atomicDisplacemants = np.random.random((len(system1),3))
        molecularDisplacemants = system1.decomposeDisplacements(atomicDisplacemants)
        for atomicDisplacemant, molecularDisplacement in zip(atomicDisplacemants, molecularDisplacemants):
            self.assertTrue(np.allclose(atomicDisplacemant, molecularDisplacement[0]))
            self.assertTrue(np.allclose(molecularDisplacement[1], np.zeros(3)))
            self.assertEqual(len(molecularDisplacement[2]), 1)
            self.assertTrue(np.allclose(molecularDisplacement[2][0], np.zeros(3)))
