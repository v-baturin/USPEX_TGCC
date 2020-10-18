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
from ..Crystal import Crystal

PATH_WITH_TESTS = os.path.dirname(os.path.abspath(__file__))


class Crystal_Test(unittest.TestCase):

    def setUp(self):
        self.xraydata = SpectrumAnalyzer.parse('{}/spectrum.txt'.format(PATH_WITH_TESTS))

        tmp = read_vasp('{}/Ba9H6.vasp'.format(PATH_WITH_TESTS))
        self.system = Crystal(symbols=tmp.get_chemical_symbols(),
                              scaled_positions=tmp.get_scaled_positions(),
                              cell=tmp.get_cell(), xraydata=self.xraydata)

    def test_xraydistance(self):
        self.assertAlmostEqual(self.system.xraydistance, 6.4739, places=4)
