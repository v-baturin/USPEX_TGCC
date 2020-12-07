"""
USPEX.Common.XRay.unittests.SpectrumAnalyzer_Test
=================================================

Class for SpectrumAnalyzer testing

.. codeauthor:: Michele Galasso <m.galasso@yandex.com>
"""

import os
import unittest

from ase.io.vasp import read_vasp
from ...Atomistic.AtomicStructure import AtomicStructure

from ..SpectrumAnalyzer import SpectrumAnalyzer

PATH_WITH_TESTS = os.path.dirname(os.path.abspath(__file__))


class SpectrumAnalyzer_Test(unittest.TestCase):
    def setUp(self):
        tmp = read_vasp('{}/Na8Cl24.vasp'.format(PATH_WITH_TESTS))
        self.system = {
            'ID': 0,
            'structure': AtomicStructure(symbols=tmp.get_chemical_symbols(),
                                         scaled_positions=tmp.get_scaled_positions(),
                                         cell=tmp.get_cell())
        }

    def test(self):
        xraydata = SpectrumAnalyzer.parse('{}/spectrum.txt'.format(PATH_WITH_TESTS))
        analyzer = SpectrumAnalyzer(**xraydata)
        analyzer.analyze(self.system)
        fitness = self.system['xraydistance']
        self.assertAlmostEqual(fitness, 0.2303, places=4)
