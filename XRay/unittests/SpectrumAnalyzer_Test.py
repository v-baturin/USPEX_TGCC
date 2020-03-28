"""
USPEX.Common.XRay.unittests.SpectrumAnalyzer_Test
=================================================

Class for SpectrumAnalyzer testing

.. codeauthor:: Michele Galasso <m.galasso@yandex.com>
"""

import os
import unittest

from ase.io.vasp import read_vasp
from USPEX.Common.Atomistic.AtomicStructure import AtomicStructure

from ..SpectrumAnalyzer import SpectrumAnalyzer

PATH_WITH_TESTS = os.path.dirname(os.path.abspath(__file__))


class SpectrumAnalyzer_Test(unittest.TestCase):
    def setUp(self):
        tmp = read_vasp('{}/Na8Cl24.vasp'.format(PATH_WITH_TESTS))
        self.system = AtomicStructure(symbols=tmp.get_chemical_symbols(),
                                      scaled_positions=tmp.get_scaled_positions(),
                                      cell=tmp.get_cell())

    def test(self):
        xraydata = SpectrumAnalyzer.parse('{}/spectrum.txt'.format(PATH_WITH_TESTS))
        analyzer = SpectrumAnalyzer(**xraydata)
        fitness = analyzer[self.system]
        self.assertAlmostEqual(fitness, 1.2873, places=4)
