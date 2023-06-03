"""
USPEX.Common.XRay.unittests.PowderSpectrumAnalyzer_Test
=======================================================

Class for PowderSpectrumAnalyzer testing

.. codeauthor:: Michele Galasso <m.galasso@yandex.com>
"""

import unittest
from pathlib import Path

from ...components import AtomisticRepresentation, AtomisticPoolEntry
from ..PowderSpectrumAnalyzer import PowderSpectrumAnalyzer

PATH_WITH_TESTS = Path(__file__).parent


class SpectrumAnalyzer_Test(unittest.TestCase):
    def setUp(self):
        self.system = AtomisticRepresentation.readAtomicStructure(PATH_WITH_TESTS/'Na8Cl24.vasp')
        self.system['ID'] = 0
        self.system = AtomisticPoolEntry(**self.system)

    def test(self):
        xraydata = PowderSpectrumAnalyzer.parse(PATH_WITH_TESTS/'spectrum.txt')
        analyzer = PowderSpectrumAnalyzer(**xraydata)
        analyzer.analyze(self.system)
        fitness = self.system['powderSpectrumAnalyzer.xraydistance']
        self.assertAlmostEqual(fitness, 0.2303, places=4)
