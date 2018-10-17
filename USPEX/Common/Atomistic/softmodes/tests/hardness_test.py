import unittest


import numpy as np
import os

from ase.io.vasp import read_vasp
from ase.io.vasp import write_vasp

from lib.Atomistic.AtomisticConfig import AtomisticConfig
from lib.Atomistic.softmodes.calcHardness import calcHardness
from lib.Systems.Crystal.Crystal import Crystal


class Hardness_test(unittest.TestCase):
    '''
    
    '''
    @classmethod
    def setUpClass(cls):
        cls.CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    def test_graphite(self):
        tmp = read_vasp(self.CURRENT_DIR + '/graphite.POSCAR')
        symbols = tmp.get_chemical_symbols()
        graphite = Crystal(symbols=symbols, scaled_positions=tmp.get_scaled_positions(), cell=tmp.get_cell())
        params = {'symbols': ['C'], 'blocks': [[len(graphite)]], 'fixed': [[1, 1]]}
        config = AtomisticConfig(**params)
        H = calcHardness(config, graphite)
        print('Hardness = ' + str(H))
        assert np.isclose(H, 0.231, atol=1)

    def test_graphite2(self):
        tmp = read_vasp(self.CURRENT_DIR + '/graphite2.POSCAR')
        symbols = tmp.get_chemical_symbols()
        graphite = Crystal(symbols=symbols, scaled_positions=tmp.get_scaled_positions(), cell=tmp.get_cell())
        params = {'symbols': ['C'], 'blocks': [[len(graphite)]], 'fixed': [[1, 1]]}
        config = AtomisticConfig(**params)
        H = calcHardness(config, graphite)
        print('Hardness = ' + str(H))
        assert np.isclose(H, 0.433, atol=1)

    def test_graphite2_supercell(self):
        tmp = read_vasp(self.CURRENT_DIR + '/graphite2.POSCAR')
        symbols = tmp.get_chemical_symbols()
        graphite = Crystal(symbols=symbols, scaled_positions=tmp.get_scaled_positions(), cell=tmp.get_cell())
        graphite *= 3
        params = {'symbols': ['C'], 'blocks': [[len(graphite)]], 'fixed': [[1, 1]]}
        config = AtomisticConfig(**params)
        H = calcHardness(config, graphite)
        print('Hardness = ' + str(H))
        assert np.isclose(H, 0.433, atol=1)

    def test_graphite_1layer(self):
        tmp = read_vasp(self.CURRENT_DIR + '/graphite_1layer.POSCAR')
        symbols = tmp.get_chemical_symbols()
        graphite = Crystal(symbols=symbols, scaled_positions=tmp.get_scaled_positions(), cell=tmp.get_cell())
        params = {'symbols': ['C'], 'blocks': [[len(graphite)]], 'fixed': [[1, 1]]}
        config = AtomisticConfig(**params)
        H = calcHardness(config, graphite)
        print('Hardness = ' + str(H))
        assert np.isclose(H, 1.72, atol=1)

    def test_aluminium(self):
        tmp = read_vasp(self.CURRENT_DIR + '/al.POSCAR')
        symbols = tmp.get_chemical_symbols()
        system = Crystal(symbols=symbols, scaled_positions=tmp.get_scaled_positions(), cell=tmp.get_cell())
        params = {'symbols': ['Al'], 'blocks': [[len(system)]], 'fixed': [[1, 1]]}
        config = AtomisticConfig(**params)
        H = calcHardness(config, system)
        print('Hardness = ' + str(H))
        assert np.isclose(H, 10.562, atol=1)

    def test_diamond(self):
        tmp = read_vasp(self.CURRENT_DIR + '/diamond.POSCAR')
        symbols = tmp.get_chemical_symbols()
        diamond = Crystal(symbols=symbols, scaled_positions=tmp.get_scaled_positions(), cell=tmp.get_cell())
        params = {'symbols': ['C'], 'blocks': [[len(diamond)]], 'fixed': [[1, 1]]}
        config = AtomisticConfig(**params)
        H = calcHardness(config, diamond)
        print('Hardness = ' + str(H))
        assert np.isclose(H, 89.656, atol=1)

    def test_Mg4Al8O16(self):
        tmp = read_vasp(self.CURRENT_DIR + '/Mg4Al8O16.POSCAR')
        symbols = tmp.get_chemical_symbols()
        system = Crystal(symbols=symbols, scaled_positions=tmp.get_scaled_positions(), cell=tmp.get_cell())
        # system *= 2
        # write_vasp('tmp.POSCAR', system, direct=True, vasp5=True, sort=True)
        params = {'symbols': ['Mg', 'Al', 'O'], 'blocks': [[4, 8, 16]], 'fixed': [[1, 1]]}
        config = AtomisticConfig(**params)
        H = calcHardness(config, system)
        print('Hardness = ' + str(H))
        assert np.isclose(H, 6.177, atol=1)

    def test_Mg4Al8O16_2(self):
        tmp = read_vasp(self.CURRENT_DIR + '/Mg4Al8O16_2.POSCAR')
        symbols = tmp.get_chemical_symbols()
        system = Crystal(symbols=symbols, scaled_positions=tmp.get_scaled_positions(), cell=tmp.get_cell())
        write_vasp('tmp.vasp', system, vasp5=True, sort=True)
        params = {'symbols': ['Mg', 'Al', 'O'], 'blocks': [[4, 8, 16]], 'fixed': [[1, 1]]}
        config = AtomisticConfig(**params)
        H = calcHardness(config, system)
        print('Hardness = ' + str(H))
        assert np.isclose(H, 19.970, atol=1)
