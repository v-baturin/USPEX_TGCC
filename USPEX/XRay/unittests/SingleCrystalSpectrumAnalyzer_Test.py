"""
USPEX.Common.XRay.unittests.SingleCrystalSpectrumAnalyzer_Test
==============================================================

Class for SingleCrystalSpectrumAnalyzer testing

.. codeauthor:: Michele Galasso <m.galasso@yandex.com>
"""

import unittest

from pathlib import Path

from ...DataModel.Flavour import Flavour
from ...components import Atomistic
from ..SingleCrystalSpectrumAnalyzer import SingleCrystalSpectrumAnalyzer

PATH_WITH_TESTS = Path(__file__).parent


class SpectrumAnalyzer_Test(unittest.TestCase):
    def setUp(self):
        propertyExtensions = dict(atomistic=Atomistic().propertyExtension())
        self.system = Flavour(extensions=propertyExtensions, **Atomistic.readAtomicStructure(PATH_WITH_TESTS/'Mg4O12Si4.vasp'))

    def test(self):
        hklFile = PATH_WITH_TESTS/'test_P1.hkl'
        expReflections = SingleCrystalSpectrumAnalyzer.parse(hklFile)
        cellParameters = (4.7877, 4.9480, 6.9151, 90, 90, 90)
        analyzer = SingleCrystalSpectrumAnalyzer(expReflections, cellParameters)
        analyzer.analyze(self.system)
        fitness = self.system['singleCrystalSpectrumAnalyzer.xraydistance']
        self.assertAlmostEqual(fitness, 0.3633, places=4)
