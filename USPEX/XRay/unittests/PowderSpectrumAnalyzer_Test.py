"""
USPEX.Common.XRay.unittests.PowderSpectrumAnalyzer_Test
=======================================================

Class for PowderSpectrumAnalyzer testing

.. codeauthor:: Michele Galasso <m.galasso@yandex.com>
"""

import os
import unittest
from os.path import join as pj

from ...components import AtomisticRepresentation
from ..PowderSpectrumAnalyzer import PowderSpectrumAnalyzer

PATH_WITH_TESTS = os.path.dirname(os.path.abspath(__file__))


class SpectrumAnalyzer_Test(unittest.TestCase):
    def setUp(self):
        self.system = AtomisticRepresentation.readAtomicStructure(pj(PATH_WITH_TESTS, 'Na8Cl24.vasp'))
        self.system['ID'] = 0

    def test(self):
        xraydata = PowderSpectrumAnalyzer.parse('{}/spectrum.txt'.format(PATH_WITH_TESTS))
        analyzer = PowderSpectrumAnalyzer(**xraydata)
        analyzer.analyze(self.system)
        fitness = self.system['powderSpectrumAnalyzer.xraydistance']
        self.assertAlmostEqual(fitness, 0.2303, places=4)
