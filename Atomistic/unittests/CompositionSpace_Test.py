"""
USPEX.Common.Atomistic.unittests.Config_Test
============================================

Class for Config testing

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import unittest
import os

from ase.io.vasp import read_vasp

from ..AtomicStructure import AtomicStructure
from ..CompositionSpace import CompositionSpace

PATH_WITH_TESTS = os.path.dirname(os.path.abspath(__file__))


class CompositionSpace_Test(unittest.TestCase):

    def setUp(self):
        # data_2109-TOPOS_fmj_fmj

        tmp = read_vasp('{}/system1.vasp'.format(PATH_WITH_TESTS))
        self.system1 = AtomicStructure(symbols=tmp.get_chemical_symbols(),
                                       scaled_positions=tmp.get_scaled_positions(),
                                       cell=tmp.get_cell())
        tmp = read_vasp('{}/system2.vasp'.format(PATH_WITH_TESTS))
        self.system2 = AtomicStructure(symbols=tmp.get_chemical_symbols(),
                                       scaled_positions=tmp.get_scaled_positions(),
                                       cell=tmp.get_cell())
        tmp = read_vasp('{}/system3.vasp'.format(PATH_WITH_TESTS))
        self.system3 = AtomicStructure(symbols=tmp.get_chemical_symbols(),
                                       scaled_positions=tmp.get_scaled_positions(),
                                       cell=tmp.get_cell())
        tmp = read_vasp('{}/system4.vasp'.format(PATH_WITH_TESTS))
        self.system4 = AtomicStructure(symbols=tmp.get_chemical_symbols(),
                                       scaled_positions=tmp.get_scaled_positions(),
                                       cell=tmp.get_cell())

    def test_fixed(self):
        config = CompositionSpace(symbols=['Mg', 'Al', 'O'], blocks=[[4, 8, 16]], range=[[1, 1]])
        self.assertTrue(config.isGoodComposition(self.system1.composition))
        self.assertFalse(config.isGoodComposition(self.system2.composition))
        self.assertFalse(config.isGoodComposition(self.system3.composition))
        self.assertFalse(config.isGoodComposition(self.system4.composition))

    def test_variable(self):
        config = CompositionSpace(symbols=['Mg', 'Al', 'O'], blocks=[[1, 0, 1], [0, 2, 3]], range=[[0, 8],[0, 8]],
                                 minAt=12, maxAt=28)
        self.assertTrue(config.isGoodComposition(self.system1.composition))
        self.assertFalse(config.isGoodComposition(self.system2.composition))
        self.assertFalse(config.isGoodComposition(self.system3.composition))
        self.assertFalse(config.isGoodComposition(self.system4.composition))

