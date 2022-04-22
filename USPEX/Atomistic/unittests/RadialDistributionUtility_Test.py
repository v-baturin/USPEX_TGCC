import os
import unittest
import numpy as np
from os.path import join as pj
from ase.io.vasp import read_vasp

from ..RadialDistributionUtility import RadialDistributionUtility
from ...components import SimpleMoleculeUtility
from ..CellUtility import Cell

PATH_WITH_TESTS = os.path.dirname(os.path.abspath(__file__))


class RadialDistributionUtility_Test(unittest.TestCase):
    def setUp(self):
        self.utility = RadialDistributionUtility()
        self.systemRDU1 = {}
        self.systemRDU2 = {}
        self.systemRDU3 = {}

        simpleMoleculeUtilityt = SimpleMoleculeUtility()

        tmp1 = read_vasp(pj(PATH_WITH_TESTS, "systemRDU1.POSCAR"))
        cell = Cell(tmp1.get_cell().array, (1,1,1))
        symbols, indices = np.unique(tmp1.get_chemical_symbols(), return_inverse = True)
        all_coordinates = tmp1.get_scaled_positions()
        coordinates = {s: [] for s in symbols}
        for index, coord in zip(indices, all_coordinates):
            coordinates[symbols[index]].append([coord])
        self.systemRDU1 = simpleMoleculeUtilityt.populateStructure(cell, coordinates, None)

        tmp2 = read_vasp(pj(PATH_WITH_TESTS, "systemRDU2.POSCAR"))
        cell = Cell(tmp2.get_cell().array, (1,1,1))
        symbols, indices = np.unique(tmp2.get_chemical_symbols(), return_inverse = True)
        all_coordinates = tmp2.get_scaled_positions()
        coordinates = {s: [] for s in symbols}
        for index, coord in zip(indices, all_coordinates):
            coordinates[symbols[index]].append([coord])
        self.systemRDU2 = simpleMoleculeUtilityt.populateStructure(cell, coordinates, None)

        tmp3 = read_vasp(pj(PATH_WITH_TESTS, "systemRDU3.POSCAR"))
        cell = Cell(tmp3.get_cell().array, (1,1,1))
        symbols, indices = np.unique(tmp3.get_chemical_symbols(), return_inverse = True)
        all_coordinates = tmp3.get_scaled_positions()
        coordinates = {s: [] for s in symbols}
        for index, coord in zip(indices, all_coordinates):
            coordinates[symbols[index]].append([coord])
        self.systemRDU3 = simpleMoleculeUtilityt.populateStructure(cell, coordinates, None)

    def test_structureOrder(self):
        self.assertAlmostEqual(self.utility.structureOrder(self.systemRDU1), 0.207, places=3)
        self.assertAlmostEqual(self.utility.structureOrder(self.systemRDU2), 0.207, places=3)
        self.assertAlmostEqual(self.utility.structureOrder(self.systemRDU3), 0.169, places=3)

    def test_averageOrder(self):
        self.assertAlmostEqual(self.utility.averageOrder(self.systemRDU1), 0.22, places=2)
        self.assertAlmostEqual(self.utility.averageOrder(self.systemRDU2), 0.22, places=2)
        self.assertAlmostEqual(self.utility.averageOrder(self.systemRDU3), 0.21, places=2)

    def test_quasientropy(self):
        self.assertAlmostEqual(self.utility.quasientropy(self.systemRDU1), 0.072, places=3)
        self.assertAlmostEqual(self.utility.quasientropy(self.systemRDU2), 0.072, places=3)
        self.assertAlmostEqual(self.utility.quasientropy(self.systemRDU3), 0.166, places=3)

    def test_distance(self):
        self.assertTrue(self.utility.equal(self.systemRDU1, self.systemRDU2))
        self.assertFalse(self.utility.equal(self.systemRDU2, self.systemRDU3))
        self.assertFalse(self.utility.equal(self.systemRDU1, self.systemRDU3))
