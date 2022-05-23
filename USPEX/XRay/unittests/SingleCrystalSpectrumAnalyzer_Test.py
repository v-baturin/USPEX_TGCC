"""
USPEX.Common.XRay.unittests.SingleCrystalSpectrumAnalyzer_Test
==============================================================

Class for SingleCrystalSpectrumAnalyzer testing

.. codeauthor:: Michele Galasso <m.galasso@yandex.com>
"""

import os
import unittest
from os.path import join as pj

from ...components import AtomisticRepresentation
from ..SingleCrystalSpectrumAnalyzer import SingleCrystalSpectrumAnalyzer

PATH_WITH_TESTS = os.path.dirname(os.path.abspath(__file__))


class SpectrumAnalyzer_Test(unittest.TestCase):
    def setUp(self):
        with open(pj(PATH_WITH_TESTS, 'Mg4O12Si4.vasp'), 'rt') as f:
            self.system = AtomisticRepresentation.readAtomicStructure(f)
        self.system['ID'] = 1

    def test(self):
        hklFile = '{}/test_P1.hkl'.format(PATH_WITH_TESTS)
        expReflections = SingleCrystalSpectrumAnalyzer.parse(hklFile)
        cellParameters = (4.7877, 4.9480, 6.9151, 90, 90, 90)
        analyzer = SingleCrystalSpectrumAnalyzer(expReflections, cellParameters)
        analyzer.analyze(self.system)
        fitness = self.system['singleCrystalSpectrumAnalyzer.xraydistance']
        self.assertAlmostEqual(fitness, 0.3633, places=4)
