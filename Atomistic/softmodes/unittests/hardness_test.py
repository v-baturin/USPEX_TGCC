import unittest


import numpy as np
import os

from ase.io.vasp import read_vasp
from ase.io.vasp import write_vasp

from ...CellUtility import Cell
from ...AtomicPrimitives import AtomicStructure
from ...Element import Element
from ..calcHardness import calcHardness


def symbolsToElements(symbols):
    return [Element(s) for s in symbols]


class Hardness_test(unittest.TestCase):
    '''
    
    '''
    @classmethod
    def setUpClass(cls):
        cls.CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    def test_graphite(self):
        tmp = read_vasp(self.CURRENT_DIR + '/graphite.POSCAR')
        cell = Cell(tmp.get_cell().array, (1,1,1))
        graphite = AtomicStructure(symbolsToElements(tmp.get_chemical_symbols()),
                                 cell.fractionalToCartesian(tmp.get_scaled_positions()),
                                 cell = cell)

        # graphite = Crystal(symbols=symbols, scaled_positions=tmp.get_scaled_positions(), cell=tmp.get_cell())
        H = calcHardness(graphite)
        print('Hardness = ' + str(H))
        self.assertAlmostEqual(H, 0.231, places=3)

    def test_graphite2(self):
        tmp = read_vasp(self.CURRENT_DIR + '/graphite2.POSCAR')
        cell = Cell(tmp.get_cell().array, (1,1,1))
        graphite = AtomicStructure(symbolsToElements(tmp.get_chemical_symbols()),
                                 cell.fractionalToCartesian(tmp.get_scaled_positions()),
                                 cell = cell)
        # graphite = Crystal(symbols=symbols, scaled_positions=tmp.get_scaled_positions(), cell=tmp.get_cell())
        H = calcHardness(graphite)
        print('Hardness = ' + str(H))
        self.assertAlmostEqual(H, 0.433, places=3)

    def test_graphite2_supercell(self):
        tmp = read_vasp(self.CURRENT_DIR + '/graphite2.POSCAR')
        tmp *= 3
        cell = Cell(tmp.get_cell().array, (1,1,1))
        graphite = AtomicStructure(symbolsToElements(tmp.get_chemical_symbols()),
                                 cell.fractionalToCartesian(tmp.get_scaled_positions()),
                                 cell = cell)
        # graphite = Crystal(symbols=symbols, scaled_positions=tmp.get_scaled_positions(), cell=tmp.get_cell())
        # graphite *= 3
        H = calcHardness(graphite)
        print('Hardness = ' + str(H))
        self.assertAlmostEqual(H, 0.433, places=3)

    def test_graphite_1layer(self):
        tmp = read_vasp(self.CURRENT_DIR + '/graphite_1layer.POSCAR')
        cell = Cell(tmp.get_cell().array, (1,1,1))
        graphite = AtomicStructure(symbolsToElements(tmp.get_chemical_symbols()),
                                 cell.fractionalToCartesian(tmp.get_scaled_positions()),
                                 cell = cell)
        # graphite = Crystal(symbols=symbols, scaled_positions=tmp.get_scaled_positions(), cell=tmp.get_cell())
        H = calcHardness(graphite)
        print('Hardness = ' + str(H))
        self.assertAlmostEqual(H, 1.72, places=1)

    def test_aluminium(self):
        tmp = read_vasp(self.CURRENT_DIR + '/al.POSCAR')
        cell = Cell(tmp.get_cell().array, (1,1,1))
        system = AtomicStructure(symbolsToElements(tmp.get_chemical_symbols()),
                                 cell.fractionalToCartesian(tmp.get_scaled_positions()),
                                 cell = cell)
        # system = Crystal(symbols=symbols, scaled_positions=tmp.get_scaled_positions(), cell=tmp.get_cell())
        H = calcHardness(system)
        print('Hardness = ' + str(H))
        self.assertAlmostEqual(H, 10.562, places=3)

    def test_diamond(self):
        tmp = read_vasp(self.CURRENT_DIR + '/diamond.POSCAR')
        cell = Cell(tmp.get_cell().array, (1,1,1))
        diamond = AtomicStructure(symbolsToElements(tmp.get_chemical_symbols()),
                                 cell.fractionalToCartesian(tmp.get_scaled_positions()),
                                 cell = cell)
        # diamond = Crystal(symbols=symbols, scaled_positions=tmp.get_scaled_positions(), cell=tmp.get_cell())
        H = calcHardness(diamond)
        print('Hardness = ' + str(H))
        self.assertAlmostEqual(H, 89.656, places=3)

    def test_Mg4Al8O16(self):
        tmp = read_vasp(self.CURRENT_DIR + '/Mg4Al8O16.POSCAR')
        cell = Cell(tmp.get_cell().array, (1,1,1))
        system = AtomicStructure(symbolsToElements(tmp.get_chemical_symbols()),
                                 cell.fractionalToCartesian(tmp.get_scaled_positions()),
                                 cell = cell)
        # system = Crystal(symbols=symbols, scaled_positions=tmp.get_scaled_positions(), cell=tmp.get_cell())
        # system *= 2
        # write_vasp('tmp.POSCAR', system, direct=True, vasp5=True, sort=True)
        H = calcHardness(system)
        print('Hardness = ' + str(H))
        self.assertAlmostEqual(H, 6.177, places=3)

    def test_Mg4Al8O16_2(self):
        tmp = read_vasp(self.CURRENT_DIR + '/Mg4Al8O16_2.POSCAR')
        cell = Cell(tmp.get_cell().array, (1,1,1))
        system = AtomicStructure(symbolsToElements(tmp.get_chemical_symbols()),
                                 cell.fractionalToCartesian(tmp.get_scaled_positions()),
                                 cell = cell)
        # system = Crystal(symbols=symbols, scaled_positions=tmp.get_scaled_positions(), cell=tmp.get_cell())
        #write_vasp('tmp.vasp', system, vasp5=True, sort=True)
        H = calcHardness(system)
        print('Hardness = ' + str(H))
        self.assertAlmostEqual(H, 19.970, places=0)
