"""
USPEX.Common.XRay.unittests.SingleCrystalSpectrumAnalyzer_Test
==============================================================

Class for SingleCrystalSpectrumAnalyzer testing

.. codeauthor:: Michele Galasso <m.galasso@yandex.com>
"""

import os
import unittest
import numpy as np


from ase.io.vasp import read_vasp
from ...Atomistic.CellUtility import Cell
from ...components import SimpleMoleculeUtility

from ..SingleCrystalSpectrumAnalyzer import SingleCrystalSpectrumAnalyzer

PATH_WITH_TESTS = os.path.dirname(os.path.abspath(__file__))


class SpectrumAnalyzer_Test(unittest.TestCase):
    def setUp(self):
        tmp = read_vasp('{}/Mg4O12Si4.vasp'.format(PATH_WITH_TESTS))
        simpleMoleculeUtility = SimpleMoleculeUtility()
        cell = Cell(tmp.get_cell().array, (1, 1, 1))
        symbols, indices = np.unique(tmp.get_chemical_symbols(), return_inverse=True)
        coordinates = {s: [] for s in symbols}
        for index, coord in zip(indices, tmp.get_scaled_positions()):
            coordinates[symbols[index]].append([coord])
        self.system = {
            'ID': 1,
            'molecules': simpleMoleculeUtility.populateStructure(cell, coordinates, None),
            'cell': cell
        }

    def test(self):
        hklFile = '{}/test_P1.hkl'.format(PATH_WITH_TESTS)
        cellParameters = (4.7877, 4.9480, 6.9151, 90, 90, 90)
        analyzer = SingleCrystalSpectrumAnalyzer(hklFile, cellParameters)
        analyzer.analyze(self.system)
        fitness = self.system['singleCrystalSpectrumAnalyzer.xraydistance']
        self.assertAlmostEqual(fitness, 0.3633, places=4)
