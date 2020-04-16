"""
USPEX.Common.Atomistic.unittests.Config_Test
============================================

Class for Config testing

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import json
import os
import unittest

from ase.io.vasp import read_vasp

from ..AtomicStructure import AtomicStructure
from ..AtomisticConfig import AtomisticConfig

PATH_WITH_TESTS = os.path.dirname(os.path.abspath(__file__))


class Config_Test(unittest.TestCase):

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
        tmp = read_vasp('{}/diamond8.vasp'.format(PATH_WITH_TESTS))
        self.diamond8 = AtomicStructure(symbols=tmp.get_chemical_symbols(),
                                        scaled_positions=tmp.get_scaled_positions(),
                                        cell=tmp.get_cell())
        tmp = read_vasp('{}/dia_2x2x2.vasp'.format(PATH_WITH_TESTS))
        self.dia_2x2x2 = AtomicStructure(symbols=tmp.get_chemical_symbols(),
                                         scaled_positions=tmp.get_scaled_positions(),
                                         cell=tmp.get_cell())

    def test_fixed(self):
        config = AtomisticConfig(symbols=['Mg', 'Al', 'O'], blocks=[[4, 8, 16]], fixed=[[1, 1]])
        self.assertTrue(config.isGoodComposition(self.system1))
        self.assertFalse(config.isGoodComposition(self.system2))
        self.assertFalse(config.isGoodComposition(self.system3))
        self.assertFalse(config.isGoodComposition(self.system4))

    def test_variable(self):
        config = AtomisticConfig(symbols=['Mg', 'Al', 'O'], blocks=[[1, 0, 1], [0, 2, 3]], fixed=[[0, 8],[0, 8]],
                                 minAt=12, maxAt=28)
        self.assertTrue(config.isGoodComposition(self.system1))
        self.assertFalse(config.isGoodComposition(self.system2))
        self.assertFalse(config.isGoodComposition(self.system3))
        self.assertFalse(config.isGoodComposition(self.system4))

    def test_problem_1(self):
        config = AtomisticConfig(symbols=['Si'], blocks=[[8]], fixed=[[1, 1]])
        self.assertTrue(config.isGoodSystem(self.diamond8))

    def test_problem_2(self):
        config = AtomisticConfig(symbols=['Si'], blocks=[[16]], fixed=[[1, 1]])
        self.assertTrue(config.isGoodSystem(self.dia_2x2x2))

    def test_serialization(self):
        config = AtomisticConfig(symbols=['Si'], blocks=[[16]], fixed=[[1, 1]])
        filepath = f"{PATH_WITH_TESTS}/tmp_config"
        with open(filepath, 'w') as fp:
            json.dump(config.toDICT(), fp)
        with open(filepath, "rt") as f:
            config1 = AtomisticConfig.fromDICT(json.loads(f.read()))
        os.remove(filepath)
