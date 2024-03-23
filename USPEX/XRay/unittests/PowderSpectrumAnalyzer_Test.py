"""
USPEX.Common.XRay.unittests.PowderSpectrumAnalyzer_Test
=======================================================

Class for PowderSpectrumAnalyzer testing

.. codeauthor:: Michele Galasso <m.galasso@yandex.com>
"""

import unittest
from pathlib import Path

from ...DataModel.Flavour import Flavour
from ...components import Atomistic
from ..PowderSpectrumAnalyzer import PowderSpectrumAnalyzer

PATH_WITH_TESTS = Path(__file__).parent


class SpectrumAnalyzer_Test(unittest.TestCase):
    def setUp(self):
        propertyExtensions = dict(atomistic=Atomistic().propertyExtension())
        self.system = Flavour(extensions=propertyExtensions, **Atomistic.readAtomicStructure(PATH_WITH_TESTS/'Na8Cl24.vasp'))

    def test(self):
        xraydata = PowderSpectrumAnalyzer.parse(PATH_WITH_TESTS/'spectrum.txt')
        analyzer = PowderSpectrumAnalyzer(**xraydata)
        analyzer.analyze(self.system)
        fitness = self.system['powderSpectrumAnalyzer.xraydistance']
        self.assertAlmostEqual(fitness, 0.2303, places=4)
