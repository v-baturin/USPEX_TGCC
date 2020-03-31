import numpy as np
import os
import unittest

from ase.io.vasp import read_vasp
from os.path import join as pj

from ...AtomisticConfig import AtomisticConfig
from ...AtomicStructure import AtomicStructure
from ..Softmodes import Softmodes
from ...AtomicStructure import AtomicStructure
from ...Crystal import Crystal


class SoftModes_test(unittest.TestCase):

    # MgAlO-systems


    def setUp(self):
        self.CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    def test_MgAlO_system_1(self):
        tmp = read_vasp(pj(self.CURRENT_DIR, 'MgAlO_system_1.vasp'))
        system = Crystal(scaled_positions=tmp.get_scaled_positions(),
                         cell=tmp.get_cell(),
                         symbols=tmp.get_chemical_symbols())
        params = {'symbols': ['Mg', 'Al', 'O'], 'blocks': [[4, 8, 16]], 'fixed': [[1, 1]]}
        config = AtomisticConfig(**params)
        softmodes = Softmodes(config, system)

    def test_MgAlO_system_2(self):
        tmp = read_vasp(pj(self.CURRENT_DIR, 'MgAlO_system_2.vasp'))
        system = Crystal(scaled_positions=tmp.get_scaled_positions(),
                         cell=tmp.get_cell(),
                         symbols=tmp.get_chemical_symbols())
        N = len(system)
        params = {'symbols': ['Mg', 'Al', 'O'], 'blocks': [[4, 8, 16]], 'fixed': [[1, 1]]}
        config = AtomisticConfig(**params)
        softmodes = Softmodes(config, system)


    def test_MgAlO_system_3(self):
        tmp = read_vasp(pj(self.CURRENT_DIR, 'MgAlO_system_3.vasp'))
        system = Crystal(scaled_positions=tmp.get_scaled_positions(),
                         cell=tmp.get_cell(),
                         symbols=tmp.get_chemical_symbols())
        N = len(system)
        params = {'symbols': ['Mg', 'Al', 'O'], 'blocks': [[4, 8, 16]], 'fixed': [[1, 1]]}
        config = AtomisticConfig(**params)
        softmodes = Softmodes(config, system)


    # Carbon systems
    def test_graphite(self):
        tmp = read_vasp('{}/graphite.POSCAR'.format(self.CURRENT_DIR))
        symbols = tmp.get_chemical_symbols()
        graphite = AtomicStructure(symbols=symbols, scaled_positions=tmp.get_scaled_positions(), cell=tmp.get_cell())
        N = len(graphite)
        params = {'symbols' : ['C'], 'blocks' : [[N]], 'fixed' : [[1, 1]]}
        config = AtomisticConfig(**params)
        softModes = Softmodes(config, graphite)

        freq_ref = [-3.14018492e-16, -2.07235140e-17, 0.00000000e+00, 0.00000000e+00, 2.74458612e-23, 7.54840750e-17,
                     1.78390610e-16, 8.28730183e-03, 2.70874463e+00,  2.70874473e+00,  2.70892967e+00,  2.70893071e+00]

        assert np.isclose([mode.frequency for mode in softModes], freq_ref).all()

    def test_graphite_supercell(self):
        tmp = read_vasp('{}/graphite.POSCAR'.format(self.CURRENT_DIR))
        symbols = tmp.get_chemical_symbols()
        graphite = Crystal(symbols=symbols, scaled_positions=tmp.get_scaled_positions(), cell=tmp.get_cell())
        graphite *= 2
        N = len(graphite)
        params = {'symbols' : ['C'], 'blocks' : [[N]], 'fixed' : [[1,1]]}
        config = AtomisticConfig(**params)
        softModes = Softmodes(config, graphite)
        freq_ref = [ -6.88734374e-16 , -5.81698639e-16 , -5.81698639e-16 , -3.31208915e-16,
                     -2.44283384e-16 , -2.44283384e-16 , -2.00449320e-16 , -2.00449320e-16,
                     -1.84332342e-16 , -1.84332342e-16 , -8.40564574e-17 , -8.40564574e-17,
                      0.00000000e+00 ,  0.00000000e+00 ,  0.00000000e+00 ,  0.00000000e+00,
                      0.00000000e+00 ,  0.00000000e+00 ,  0.00000000e+00 ,  0.00000000e+00,
                      0.00000000e+00 ,  0.00000000e+00 ,  0.00000000e+00 ,  0.00000000e+00,
                      0.00000000e+00 ,  0.00000000e+00 ,  0.00000000e+00 ,  0.00000000e+00,
                      2.41937007e-17 ,  2.41937007e-17 ,  1.32307517e-16 ,  1.32307517e-16,
                      1.52013694e-16 ,  1.52013694e-16 ,  2.21168035e-16 ,  2.73064983e-16,
                      2.73064983e-16 ,  4.19113190e-16 ,  4.19113190e-16 ,  5.46889927e-16,
                      4.14365091e-03 ,  4.14365091e-03 ,  4.14365091e-03 ,  4.14365091e-03,
                      4.14365091e-03 ,  4.14365091e-03 ,  4.14365091e-03 ,  4.14365091e-03,
                      8.28730183e-03 ,  8.28730183e-03 ,  8.28730183e-03 ,  8.28730183e-03,
                      9.02942371e-01 ,  9.02942371e-01 ,  9.02942412e-01 ,  9.02942412e-01,
                      9.02943911e-01 ,  9.02943911e-01 ,  9.02943911e-01 ,  9.02943911e-01,
                      9.02949376e-01 ,  9.02949376e-01 ,  9.02952882e-01 ,  9.02952882e-01,
                      1.80584705e+00 ,  1.80584705e+00 ,  1.80584705e+00 ,  1.80584705e+00,
                      1.80584812e+00 ,  1.80584812e+00 ,  1.80584939e+00 ,  1.80584939e+00,
                      1.80597783e+00 ,  1.80597783e+00 ,  1.80598030e+00 ,  1.80598030e+00,
                      2.70874463e+00 ,  2.70874463e+00 ,  2.70874463e+00 ,  2.70874463e+00,
                      2.70874473e+00 ,  2.70874473e+00 ,  2.70874473e+00 ,  2.70874473e+00,
                      2.70888344e+00 ,  2.70888344e+00 ,  2.70888344e+00 ,  2.70888344e+00,
                      2.70888357e+00 ,  2.70888357e+00 ,  2.70888481e+00 ,  2.70888481e+00,
                      2.70892967e+00 ,  2.70892967e+00 ,  2.70893071e+00 ,  2.70893071e+00]

        assert np.isclose([mode.frequency for mode in softModes], freq_ref).all()

    def test_graphite2(self):
        tmp = read_vasp('{}/graphite2.POSCAR'.format(self.CURRENT_DIR))
        symbols = tmp.get_chemical_symbols()
        graphite = AtomicStructure(symbols=symbols, scaled_positions=tmp.get_scaled_positions(), cell=tmp.get_cell())
        N = len(graphite)
        params = {'symbols' : ['C'], 'blocks' : [[N]], 'fixed' : [[1,1]]}
        config = AtomisticConfig(**params)
        softModes = Softmodes(config, graphite)

        freq_ref = [-3.14018492e-16, -2.07235140e-17, 0.00000000e+00, 0.00000000e+00, 2.74458612e-23, 7.54840750e-17,
                     1.78390610e-16, 8.28730183e-03, 2.70874463e+00,  2.70874473e+00,  2.70892967e+00,  2.70893071e+00]

        assert np.isclose([mode.frequency for mode in softModes], freq_ref).all()

    def test_graphite2_supercell(self):
        tmp = read_vasp('{}/graphite2.POSCAR'.format(self.CURRENT_DIR))
        symbols = tmp.get_chemical_symbols()
        graphite = AtomicStructure(symbols=symbols, scaled_positions=tmp.get_scaled_positions(), cell=tmp.get_cell())
        graphite *= 2
        N = len(graphite)
        params = {'symbols' : ['C'], 'blocks' : [[N]], 'fixed' : [[1,1]]}
        config = AtomisticConfig(**params)
        softModes = Softmodes(config, graphite)

        freq_ref = [-3.14018492e-16, -2.07235140e-17, 0.00000000e+00, 0.00000000e+00, 2.74458612e-23, 7.54840750e-17,
                     1.78390610e-16, 8.28730183e-03, 2.70874463e+00,  2.70874473e+00,  2.70892967e+00,  2.70893071e+00]

        assert np.isclose([mode.frequency for mode in softModes], freq_ref).all()

    def test_diamond(self):
        tmp = read_vasp('{}/diamond.POSCAR'.format(self.CURRENT_DIR))
        symbols = tmp.get_chemical_symbols()
        diamond = AtomicStructure(symbols=symbols, scaled_positions=tmp.get_scaled_positions(), cell=tmp.get_cell())
        N = len(diamond)
        params = {'symbols' : ['C'], 'blocks' : [[N]], 'fixed' : [[1,1]]}
        config = AtomisticConfig(**params)

        softModes = Softmodes(config, diamond)
        freq = [mode.frequency for mode in softModes]
        freq_ref = [ -2.22044605e-16, -2.22044605e-16, -2.22044605e-16,  1.66088535e+00, 1.66088535e+00,  1.66088535e+00]
        assert np.isclose(freq, freq_ref).all()

    def test_diamond_supercell(self):
        tmp = read_vasp('{}/diamond.POSCAR'.format(self.CURRENT_DIR))
        symbols = tmp.get_chemical_symbols()
        diamond = AtomicStructure(symbols=symbols, scaled_positions=tmp.get_scaled_positions(), cell=tmp.get_cell())
        diamond *= 2
        N = len(diamond)
        params = {'symbols' : ['C'], 'blocks' : [[N]], 'fixed' : [[1,1]]}
        config = AtomisticConfig(**params)
        softModes = Softmodes(config, diamond)

        freq_ref = [ -4.81633103e-16,  -2.68342179e-16 , -2.38233999e-16 , -2.38233999e-16,
                     -2.07559197e-16,  -2.07559197e-16 , -9.21082950e-17 , -9.21082950e-17,
                     -6.07025264e-17,  -6.07025264e-17 , -3.91132036e-17 ,  1.09576306e-16,
                      2.17278827e-16,   2.17278827e-16 ,  2.69936911e-16 ,  5.45950670e-16,
                      5.45950670e-16,   4.15221337e-01 ,  4.15221337e-01 ,  4.15221337e-01,
                      4.15221337e-01,   8.30442674e-01 ,  8.30442674e-01 ,  8.30442674e-01,
                      8.30442674e-01,   8.30442674e-01 ,  8.30442674e-01 ,  1.24566401e+00,
                      1.24566401e+00,   1.24566401e+00 ,  1.24566401e+00 ,  1.66088535e+00,
                      1.66088535e+00,   1.66088535e+00 ,  1.66088535e+00 ,  1.66088535e+00,
                      1.66088535e+00,   1.66088535e+00 ,  1.66088535e+00 ,  1.66088535e+00,
                      1.66088535e+00,   1.66088535e+00 ,  1.66088535e+00 ,  1.66088535e+00,
                      1.66088535e+00,   1.66088535e+00 ,  1.66088535e+00 ,  1.66088535e+00]
        assert np.isclose([mode.frequency for mode in softModes], freq_ref).all()

    # MgO-systems

    def test_MgO_1(self):
        symbols = 4*['Mg'] + 4*['O']
        cell = [[ 4.257976  ,   0.000000 ,    0.000000],
                [ 0.000000   ,  4.257976  ,   0.000000],
                [ 0.000000   ,  0.000000  ,   4.257977]]
        scaled_positions = [[0.0, 0.0, 0.0],
                            [0.5, 0.5, 0.0],
                            [0.0, 0.5, 0.5],
                            [0.5, 0.0, 0.5],
                            [0.5, 0.0, 0.0],
                            [0.0, 0.5, 0.0],
                            [0.0, 0.0, 0.5],
                            [0.5, 0.5, 0.5]]
        system = AtomicStructure(scaled_positions=scaled_positions, cell=cell, symbols=symbols, pbc=True)

        params = {'symbols': ['Mg', 'O'], 'blocks': [[4, 4]], 'fixed': [[1, 1]]}
        config = AtomisticConfig(**params)

        softmodes = Softmodes(config, system)
        print('MgO_1')


    def test_MgO_2(self):
        symbols = ['Mg', 'O']
        cell = [[    3.009788   ,  0.000000   ,  0.000000],
                [    1.504894   ,  2.606553   ,  0.000000],
                [    1.504894   ,  0.868851   ,  2.457482]]

        scaled_positions = [[0.0, 0.0, 0.0],
                            [0.5, 0.5, 0.5]]
        system = AtomicStructure(scaled_positions=scaled_positions, cell=cell, symbols=symbols, pbc=True)

        params = {'symbols': ['Mg', 'O'], 'blocks': [[1, 1]], 'fixed': [[1, 1]]}
        config = AtomisticConfig(**params)

        softmodes = Softmodes(config, system)
        print('MgO_1')
