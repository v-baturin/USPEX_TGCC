"""
USPEX.Common.XRay.unittests.PowderSpectrumAnalyzer_Test
=======================================================

Class for PowderSpectrumAnalyzer testing

.. codeauthor:: Michele Galasso <m.galasso@yandex.com>
"""

import os
import unittest
import numpy as np

from ase.io.vasp import read_vasp
from ...Atomistic.CellUtility import Cell
from ...components import SimpleMoleculeUtility

from ..PowderSpectrumAnalyzer import PowderSpectrumAnalyzer

PATH_WITH_TESTS = os.path.dirname(os.path.abspath(__file__))


class SpectrumAnalyzer_Test(unittest.TestCase):
    def setUp(self):
        tmp = read_vasp('{}/Na8Cl24.vasp'.format(PATH_WITH_TESTS))
        simpleMoleculeUtility = SimpleMoleculeUtility()
        cell = Cell(tmp.get_cell().array, (1, 1, 1))
        symbols, indices = np.unique(tmp.get_chemical_symbols(), return_inverse=True)
        coordinates = {s: [] for s in symbols}
        for index, coord in zip(indices, tmp.get_scaled_positions()):
            coordinates[symbols[index]].append([coord])
        self.system = {
            'ID': 0,
            'molecules': simpleMoleculeUtility.populateStructure(cell, coordinates, None),
            'cell': cell
        }

    def test(self):
        xraydata = PowderSpectrumAnalyzer.parse('{}/spectrum.txt'.format(PATH_WITH_TESTS))
        analyzer = PowderSpectrumAnalyzer(**xraydata)
        analyzer.analyze(self.system)
        fitness = self.system['powderSpectrumAnalyzer.xraydistance']
        self.assertAlmostEqual(fitness, 0.2303, places=4)
