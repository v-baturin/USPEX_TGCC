"""
USPEX.Common.Atomistic.unittests.CrystalConfig_Test
===================================================

Class for CrystalConfig testing

.. codeauthor:: Michele Galasso <m.galasso@yandex.com>
"""

import unittest
import os

from ase.io.vasp import read_vasp

from ...XRay.SpectrumAnalyzer import SpectrumAnalyzer
from ..AtomicStructure import AtomicStructure
from ..CrystalConfig import CrystalConfig

PATH_WITH_TESTS = os.path.dirname(os.path.abspath(__file__))


class CrystalConfig_Test(unittest.TestCase):

    def setUp(self):
        self.xraydata = SpectrumAnalyzer.parse('{}/spectrum.txt'.format(PATH_WITH_TESTS))

        tmp = read_vasp('{}/Ba9H6.vasp'.format(PATH_WITH_TESTS))
        self.system = AtomicStructure(symbols=tmp.get_chemical_symbols(),
                                      scaled_positions=tmp.get_scaled_positions(),
                                      cell=tmp.get_cell())

    def test_xraydistance(self):
        config = CrystalConfig(symbols=['Ba', 'H'], blocks=[[1, 0], [0, 1]], fixed=[[0, 18], [0, 18]],
                               minAt=8, maxAt=18, xraydata=self.xraydata)
        self.assertAlmostEqual(config.xraydistance(self.system), 9.3398, places=4)
