"""
USPEX.Common.Atomistic.unittests.Config_Test
============================================

Class for Config testing

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import unittest
from pathlib import Path

from ...components import AtomisticRepresentation, CompositionSpace, SimpleMoleculeUtility, AtomisticPoolEntry

PATH_WITH_TESTS = Path(__file__).parent


class CompositionSpace_Test(unittest.TestCase):

    def setUp(self):
        # data_2109-TOPOS_fmj_fmj

        self.simpleMoleculeUtility = SimpleMoleculeUtility()
        self.system1 = AtomisticPoolEntry(**AtomisticRepresentation.readAtomicStructure(PATH_WITH_TESTS/'system1.vasp'))
        self.system2 = AtomisticPoolEntry(**AtomisticRepresentation.readAtomicStructure(PATH_WITH_TESTS/'system2.vasp'))
        self.system3 = AtomisticPoolEntry(**AtomisticRepresentation.readAtomicStructure(PATH_WITH_TESTS/'system3.vasp'))
        self.system4 = AtomisticPoolEntry(**AtomisticRepresentation.readAtomicStructure(PATH_WITH_TESTS/'system4.vasp'))

    def test_fixed(self):
        config = CompositionSpace(symbols=['Mg', 'Al', 'O'], blocks=[[4, 8, 16]], range=[[1, 1]])
        self.assertTrue(config.isGoodComposition(self.simpleMoleculeUtility.composition(self.system1)))
        self.assertFalse(config.isGoodComposition(self.simpleMoleculeUtility.composition(self.system2)))
        self.assertFalse(config.isGoodComposition(self.simpleMoleculeUtility.composition(self.system3)))
        self.assertFalse(config.isGoodComposition(self.simpleMoleculeUtility.composition(self.system4)))

    def test_variable(self):
        config = CompositionSpace(symbols=['Mg', 'Al', 'O'], blocks=[[1, 0, 1], [0, 2, 3]], range=[[0, 8],[0, 8]],
                                 minAt=12, maxAt=28)
        self.assertTrue(config.isGoodComposition(self.simpleMoleculeUtility.composition(self.system1)))
        self.assertFalse(config.isGoodComposition(self.simpleMoleculeUtility.composition(self.system2)))
        self.assertFalse(config.isGoodComposition(self.simpleMoleculeUtility.composition(self.system3)))
        self.assertFalse(config.isGoodComposition(self.simpleMoleculeUtility.composition(self.system4)))

