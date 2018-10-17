import unittest

import numpy as np
from ase.io.vasp import read_vasp

from USPEX.Atomistic.Private.AtomisticConfigPrivate import AtomisticConfigPrivate
from USPEX.Atomistic.Private.softmodes.Softmodes import Softmodes
from USPEX.Common.Atomistic.AtomicStructure import AtomicStructure


class SoftModes_test(unittest.TestCase):

    # MgAlO-systems

    def test_MgAlO_system_1(self):
        coords = [[0.238792900000000  , 0.740520900000000  , 0.667830000000000],
                [0.738793000000000  , 0.471855500000000  , 0.167830000000000],
                [0.238793000000000  , 0.240520900000000  , 0.167830000000000],
                [0.738792900000000  , 0.971855500000000  , 0.667830000000000],
                [0.565340500000000  , 0.724568100000000  , 0.167830000000000],
                [0.412245500000000  , 0.487808300000000  , 0.667830000000000],
                [0.0653404000000000 , 0.487808300000000  , 0.667830000000000],
                [0.912245500000000  , 0.724568100000000  , 0.167830000000000],
                [0.912245500000000  , 0.224568100000000  , 0.667830000000000],
                [0.0653404000000000 , 0.987808300000000  , 0.167830000000000],
                [0.412245500000000  , 0.987808300000000  , 0.167830000000000],
                [0.565340400000000  , 0.224568100000000  , 0.667830000000000],
                [0.488793000000000  , 0.356188200000000  , 0.167830000000000],
                [0.488792900000000  , 0.856188200000000  , 0.667830000000000],
                [0.988793000000000  , 0.856188200000000  , 0.667830000000000],
                [0.988792900000000  , 0.356188200000000  , 0.167830000000000],
                [0.597533600000000  , 0.587390500000000  , 0.667830000000000],
                [0.380052300000000  , 0.624986000000000  , 0.167830000000000],
                [0.738792900000000  , 0.806748900000000  , 0.167830000000000],
                [0.238793000000000  , 0.405627600000000  , 0.667830000000000],
                [0.880052300000000  , 0.0873905000000000 , 0.167830000000000],
                [0.0975336000000000 , 0.124986000000000  , 0.667830000000000],
                [0.738792900000000  , 0.306748900000000  , 0.667830000000000],
                [0.238793000000000  , 0.905627600000000  , 0.167830000000000],
                [0.880052300000000  , 0.587390500000000  , 0.667830000000000],
                [0.0975336000000000 , 0.624986000000000  , 0.167830000000000],
                [0.597533600000000  , 0.0873905000000000 , 0.167830000000000],
                [0.380052300000000  , 0.124986000000000  , 0.667830000000000]]
        cell = [[8.71834000000000  ,  0  , 0],
                [0 ,   8.75564100000000  ,  0],
                [0  ,  0  ,  2.63842800000000]]
        chemicalSymbols = ['Mg'] * 4 + ['Al'] * 8 + ['O'] * 16
        system = AtomicStructure(scaled_positions=coords, cell=cell, symbols=chemicalSymbols)
        N = len(system)
        params = {'symbols': ['Mg', 'Al', 'O'], 'blocks': [[4, 8, 16]], 'fixed': [[1, 1]]}
        config = AtomisticConfigPrivate(**params)
        softmodes = Softmodes(config, system)

    def test_MgAlO_new_1(self):
        cell = [[5.33332300000000 ,  0  , 0],
                [1.29439479190448 ,  4.59444298831747 ,   0],
                [-1.33982421702407,  -0.0145756339008623, 8.28570550370264]]
        scaled_positions = [[0.4273819  , 0.4169689 ,  0.9961479],
                            [0.7802706  , 0.3254624 ,  0.3365667],
                            [0.1237048  , 0.9129726 ,  0.8229303],
                            [0.8074686  , 0.8263651 ,  0.4853215],
                            [0.2699683  , 0.3808253 ,  0.3190440],
                            [0.9269368  , 0.4117413 ,  0.9967983],
                            [0.3049355  , 0.8799713 ,  0.5041435],
                            [0.6330099  , 0.9161924 ,  0.8278458],
                            [0.9797116  , 0.8917419 ,  0.1600866],
                            [0.4855877  , 0.8787274 ,  0.1622838],
                            [0.6122549  , 0.3801172 ,  0.6536809],
                            [0.1076103  , 0.3951207 ,  0.6668135],
                            [0.4818512  , 0.03276740 , 0.3675936],
                            [0.9390580  , 0.7740038 ,  0.9601472],
                            [0.06917410 , 0.4451103 ,  0.4584085],
                            [0.7063086  , 0.2676544 ,  0.8651624],
                            [0.2972952  , 0.6237449 ,  0.1821616],
                            [0.6645835  , 0.1358166 ,  0.1410745],
                            [0.4247415  , 0.1061460 ,  0.6515346],
                            [0.7918194  , 0.6575492 ,  0.6915289],
                            [0.4712711  , 0.7968283 ,  0.9534433],
                            [0.0385106 , 0.9397809 ,  0.3646387],
                            [0.1651310  , 0.3010994 ,  0.8654175],
                            [0.5176822  , 0.5319250 ,  0.4542980],
                            [0.30412  , 0.6500147 ,  0.6769173],
                            [0.9112982  , 0.1318755 ,  0.6224672],
                            [0.1501313  , 0.1671769 ,  0.1373762],
                            [0.8067350  , 0.6081254 ,  0.1830761]]
        chemicalSymbols = ['Mg'] * 4 + ['Al'] * 8 + ['O'] * 16
        system = AtomicStructure(scaled_positions=scaled_positions, cell=cell, symbols=chemicalSymbols, pbc=True)
        N = len(system)
        params = {'symbols': ['Mg', 'Al', 'O'], 'blocks': [[4, 8, 16]], 'fixed': [[1, 1]]}
        config = AtomisticConfigPrivate(**params)
        softmodes = Softmodes(config, system)

    def test_MgAlO_system(self):
        coords = [[0.35347400000000001, 0.62385400000000002, 0.25],
                  [0.8850870999999999, 0.44307239999999998, 0.75],
                  [0.8850870999999999, 0.1238539, 0.25],
                  [0.35347390000000001, 0.94307239999999992, 0.75],
                  [0.36920350000000002, 0.069829600000000006, 0.080564899999999995],
                  [0.86935759999999995, 0.99709669999999984, 0.919435],
                  [0.86935759999999995, 0.99709669999999984, 0.58056490000000005],
                  [0.36920350000000002, 0.069829600000000006, 0.419435],
                  [0.86935759999999995, 0.56982960000000005, 0.419435],
                  [0.36920350000000002, 0.4970967, 0.58056490000000005],
                  [0.36920350000000002, 0.4970967, 0.919435],
                  [0.86935759999999995, 0.56982960000000005, 0.080564899999999995],
                  [0.20186470000000001, 0.17021710000000001, 0.89277799999999996],
                  [0.20186470000000001, 0.39670919999999998, 0.10722189999999999],
                  [0.20186470000000001, 0.39670919999999998, 0.39277800000000002],
                  [0.20186470000000001, 0.17021710000000001, 0.60722189999999998],
                  [0.036696399999999997, 0.67021710000000001, 0.60722189999999998],
                  [0.036696399999999997, 0.89670919999999998, 0.39277800000000002],
                  [0.036696399999999997, 0.89670919999999998, 0.10722189999999999],
                  [0.036696399999999997, 0.67021710000000001, 0.89277799999999996],
                  [0.75160570000000004, 0.059273399999999997, 0.75],
                  [0.56156870000000003, 0.78346320000000003, 0.5],
                  [0.48695539999999998, 0.0076528999999999989, 0.25],
                  [0.56156870000000003, 0.78346320000000003, 0.0],
                  [0.75160570000000004, 0.50765289999999996, 0.25],
                  [0.67699240000000005, 0.28346320000000003, 0.0],
                  [0.48695539999999998, 0.55927340000000003, 0.75],
                  [0.67699240000000005, 0.28346320000000003, 0.5]]
        cell = [[4.5499330000000002, 0.0, 0.0],
                [2.9337473393056105e-16, 4.7911729999999997, 0.0],
                [5.7762664522789382e-16, 5.7762664522789382e-16, 9.4333589999999994]]
        chemicalSymbols = ['Mg'] * 4 + ['Al'] * 8 + ['O'] * 16
        system = AtomicStructure(scaled_positions=coords, cell=cell, symbols=chemicalSymbols)
        N = len(system)
        params = {'symbols': ['Mg', 'Al', 'O'], 'blocks': [[4, 8, 16]], 'fixed': [[1, 1]]}
        config = AtomisticConfigPrivate(**params)
        softModes = Softmodes(config, system)

    # Carbon systems

    def test_graphite(self):
        tmp = read_vasp('graphite.POSCAR')
        symbols = tmp.get_chemical_symbols()
        graphite = AtomicStructure(symbols=symbols, scaled_positions=tmp.get_scaled_positions(), cell=tmp.get_cell())
        N = len(graphite)
        params = {'symbols' : ['C'], 'blocks' : [[N]], 'fixed' : [[1, 1]]}
        config = AtomisticConfigPrivate(**params)
        softModes = Softmodes(config, graphite)

        freq_ref = [-3.14018492e-16, -2.07235140e-17, 0.00000000e+00, 0.00000000e+00, 2.74458612e-23, 7.54840750e-17,
                     1.78390610e-16, 8.28730183e-03, 2.70874463e+00,  2.70874473e+00,  2.70892967e+00,  2.70893071e+00]

        assert np.isclose([mode.frequency for mode in softModes], freq_ref).all()

    def test_graphite_supercell(self):
        tmp = read_vasp('graphite.POSCAR')
        symbols = tmp.get_chemical_symbols()
        graphite = Crystal(symbols=symbols, scaled_positions=tmp.get_scaled_positions(), cell=tmp.get_cell())
        graphite *= 2
        N = len(graphite)
        params = {'symbols' : ['C'], 'blocks' : [[N]], 'fixed' : [[1,1]]}
        config = AtomisticConfigPrivate(**params)
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
        tmp = read_vasp('graphite2.POSCAR')
        symbols = tmp.get_chemical_symbols()
        graphite = AtomicStructure(symbols=symbols, scaled_positions=tmp.get_scaled_positions(), cell=tmp.get_cell())
        N = len(graphite)
        params = {'symbols' : ['C'], 'blocks' : [[N]], 'fixed' : [[1,1]]}
        config = AtomisticConfigPrivate(**params)
        softModes = Softmodes(config, graphite)

        freq_ref = [-3.14018492e-16, -2.07235140e-17, 0.00000000e+00, 0.00000000e+00, 2.74458612e-23, 7.54840750e-17,
                     1.78390610e-16, 8.28730183e-03, 2.70874463e+00,  2.70874473e+00,  2.70892967e+00,  2.70893071e+00]

        assert np.isclose([mode.frequency for mode in softModes], freq_ref).all()

    def test_graphite2_supercell(self):
        tmp = read_vasp('graphite2.POSCAR')
        symbols = tmp.get_chemical_symbols()
        graphite = AtomicStructure(symbols=symbols, scaled_positions=tmp.get_scaled_positions(), cell=tmp.get_cell())
        graphite *= 2
        N = len(graphite)
        params = {'symbols' : ['C'], 'blocks' : [[N]], 'fixed' : [[1,1]]}
        config = AtomisticConfigPrivate(**params)
        softModes = Softmodes(config, graphite)

        freq_ref = [-3.14018492e-16, -2.07235140e-17, 0.00000000e+00, 0.00000000e+00, 2.74458612e-23, 7.54840750e-17,
                     1.78390610e-16, 8.28730183e-03, 2.70874463e+00,  2.70874473e+00,  2.70892967e+00,  2.70893071e+00]

        assert np.isclose([mode.frequency for mode in softModes], freq_ref).all()

    def test_diamond(self):
        tmp = read_vasp('diamond.POSCAR')
        symbols = tmp.get_chemical_symbols()
        diamond = AtomicStructure(symbols=symbols, scaled_positions=tmp.get_scaled_positions(), cell=tmp.get_cell())
        N = len(diamond)
        params = {'symbols' : ['C'], 'blocks' : [[N]], 'fixed' : [[1,1]]}
        config = AtomisticConfigPrivate(**params)

        softModes = Softmodes(config, diamond)
        freq = [mode.frequency for mode in softModes]
        freq_ref = [ -2.22044605e-16, -2.22044605e-16, -2.22044605e-16,  1.66088535e+00, 1.66088535e+00,  1.66088535e+00]
        assert np.isclose(freq, freq_ref).all()

    def test_diamond_supercell(self):
        tmp = read_vasp('diamond.POSCAR')
        symbols = tmp.get_chemical_symbols()
        diamond = AtomicStructure(symbols=symbols, scaled_positions=tmp.get_scaled_positions(), cell=tmp.get_cell())
        diamond *= 2
        N = len(diamond)
        params = {'symbols' : ['C'], 'blocks' : [[N]], 'fixed' : [[1,1]]}
        config = AtomisticConfigPrivate(**params)
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
        config = AtomisticConfigPrivate(**params)

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
        config = AtomisticConfigPrivate(**params)

        softmodes = Softmodes(config, system)
        print('MgO_1')
