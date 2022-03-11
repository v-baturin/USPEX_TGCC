"""
USPEX.Common.Atomistic.unittests.Config_Test
============================================

Class for Config testing

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import unittest
import os

from ...components import AtomisticRepresentation, CompositionSpace, SimpleMoleculeUtility

PATH_WITH_TESTS = os.path.dirname(os.path.abspath(__file__))


class CompositionSpace_Test(unittest.TestCase):

    def setUp(self):
        # data_2109-TOPOS_fmj_fmj

        self.simpleMoleculeUtility = SimpleMoleculeUtility()
        self.system1 = AtomisticRepresentation.readAtomicStructure('{}/system1.vasp'.format(PATH_WITH_TESTS))
        self.system2 = AtomisticRepresentation.readAtomicStructure('{}/system2.vasp'.format(PATH_WITH_TESTS))
        self.system3 = AtomisticRepresentation.readAtomicStructure('{}/system3.vasp'.format(PATH_WITH_TESTS))
        self.system4 = AtomisticRepresentation.readAtomicStructure('{}/system4.vasp'.format(PATH_WITH_TESTS))

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

