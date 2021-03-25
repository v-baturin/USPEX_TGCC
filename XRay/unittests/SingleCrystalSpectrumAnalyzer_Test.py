"""
USPEX.Common.XRay.unittests.SingleCrystalSpectrumAnalyzer_Test
==============================================================

Class for SingleCrystalSpectrumAnalyzer testing

.. codeauthor:: Michele Galasso <m.galasso@yandex.com>
"""

import os
import unittest

from ase.io.vasp import read_vasp
from ...Atomistic.AtomicStructure import AtomicStructure

from ..SingleCrystalSpectrumAnalyzer import SingleCrystalSpectrumAnalyzer

PATH_WITH_TESTS = os.path.dirname(os.path.abspath(__file__))


class SpectrumAnalyzer_Test(unittest.TestCase):
    def setUp(self):
        tmp = read_vasp('{}/Mg4O12Si4.vasp'.format(PATH_WITH_TESTS))
        self.system = {
            'ID': 1,
            'structure': AtomicStructure(symbols=tmp.get_chemical_symbols(),
                                         scaled_positions=tmp.get_scaled_positions(),
                                         cell=tmp.get_cell())
        }

    def test(self):
        xraydata = SingleCrystalSpectrumAnalyzer.parse('{}/test_P1.hkl'.format(PATH_WITH_TESTS))
        analyzer = SingleCrystalSpectrumAnalyzer(**xraydata)
        analyzer.analyze(self.system)
        fitness = self.system['singleCrystalSpectrumAnalyzer.xraydistance']
        self.assertAlmostEqual(fitness, 0.3633, places=4)
