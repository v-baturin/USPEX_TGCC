from ...AtomicStructure import AtomicStructure
from ...Bonds import Bond, Bonds
from ..BondHardness import BondHardness
from ..BondHardness_new import BondHardness_new
from ...Crystal import Crystal


from ase.io.vasp import read_vasp

import os
import unittest
import numpy as np


class test_BondHardness(unittest.TestCase):
    '''
    
    '''


    @classmethod
    def setUpClass(cls):
        cls.CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    def test_diamond(self):
        tmp = read_vasp('{}/diamond.POSCAR'.format(self.CURRENT_DIR))
        system = AtomicStructure(symbols=tmp.get_chemical_symbols(),
                                 scaled_positions=tmp.get_scaled_positions(),
                                 cell=tmp.get_cell(),
                                 pbc=True)
        goodBonds = np.array([[0.5]])
        bond_in = BondHardness(system, goodBonds)





    def test_1(self):
        scaled_positions = [[0., 0., 0.], [0.33333, 0.66667, 0.], [0., 0., 0.5], [0.66667, 0.33334, 0.5]]
        cell = [[2.456, 0., 0.], [-1.228, 2.126958, 0.], [0., 0., 6.696]]
        system = AtomicStructure(symbols=4 * ['C'], scaled_positions=scaled_positions, cell=cell, pbc=True)
        goodBonds = np.array([[0.5]])
        # params = {'symbols' : ['C'], 'blocks' : [[32]], 'fixed' : [[1,1]],
        #           'goodBonds': np.array([[0.5]]), 'minDistMatrice': np.array([[0.8]]),
        #           'minAngle': 0, 'minDiagAngle': 0, 'minVectorLength': 0,}
        # config = CrystalConfig(**params)


        bond_in_ref = Bonds()
        bond_in_ref.append(Bond(2, 3, -0.10204218, 1, [-1., -1., 0.]))
        bond_in_ref.append(Bond(0, 1, -0.10204198, 1, [0., -1., 0.]))
        bond_in_ref.append(Bond(0, 1, -0.10202091, 1, [0., 0., 0.]))
        bond_in_ref.append(Bond(2, 3, -0.10202071, 1, [0., 0., 0.]))
        bond_in_ref.append(Bond(2, 3, -0.10202071, 1, [-1., 0., 0.]))
        bond_in_ref.append(Bond(0, 1, -0.10202071, 1, [-1., -1., 0.]))
        bond_in_ref.append(Bond(0, 2, 1.828, 4, [0., 0., -1.]))
        bond_in_ref.append(Bond(0, 2, 1.828, 4, [0., 0., 0.]))

        bond_in = BondHardness(system, goodBonds)
        assert bond_in == bond_in_ref
        print()

    def test_2(self):
        cell = [[  4.912, 0., 0. ], [ -2.456, 4.253916, 0.], [0., 0., 13.392]]
        scaled_positions = [[ 0. , 0.  , 0.],
            [ 0. ,  0.5,   0. ],
            [ 0.5,  0. ,   0. ],
            [ 0.5,  0.5,   0. ],
            [ 0. ,  0. ,   0.5],
            [ 0. ,  0.5,   0.5],
            [ 0.5,  0. ,   0.5],
            [ 0.5,  0.5,   0.5],
            [ 0.166665 , 0.333335 , 0.      ],
            [ 0.166665 , 0.833335 , 0.      ],
            [ 0.666665 , 0.333335 , 0.      ],
            [ 0.666665 , 0.833335 , 0.      ],
            [ 0.166665 , 0.333335 , 0.5     ],
            [ 0.166665 , 0.833335 , 0.5     ],
            [ 0.666665 , 0.333335 , 0.5     ],
            [ 0.666665 , 0.833335 , 0.5     ],
            [ 0.  ,  0.   , 0.25],
            [ 0.  ,  0.5  , 0.25],
            [ 0.5 ,  0.   , 0.25],
            [ 0.5 ,  0.5  , 0.25],
            [ 0.  ,  0.   , 0.75],
            [ 0.  ,  0.5  , 0.75],
            [ 0.5 ,  0.   , 0.75],
            [ 0.5 ,  0.5  , 0.75],
            [ 0.333335 , 0.16667 ,  0.25    ],
            [ 0.333335 , 0.66667 ,  0.25    ],
            [ 0.833335 , 0.16667 ,  0.25    ],
            [ 0.833335 , 0.66667 ,  0.25    ],
            [ 0.333335 , 0.16667 ,  0.75    ],
            [ 0.333335 , 0.66667 ,  0.75    ],
            [ 0.833335 , 0.16667 ,  0.75    ],
            [ 0.833335 , 0.66667 ,  0.75    ]]

        system = Crystal(symbols=32 * ['C'], scaled_positions=scaled_positions, cell=cell, pbc=True)
        goodBonds = np.array([[0.5]])
        # params = {'symbols' : ['C'], 'blocks' : [[32]], 'fixed' : [[1,1]],
        #           'goodBonds': np.array([[0.5]]), 'minDistMatrice': np.array([[0.8]]),
        #           'minAngle': 0, 'minDiagAngle': 0, 'minVectorLength': 0}
        # config = CrystalConfig(**params)

        bond_in_ref = Bonds()

        bond_in_ref.append(Bond(23, 28, -0.10204218, 1,  [0., 0., 0.]))
        bond_in_ref.append(Bond(19, 24, -0.10204218, 1,  [      0. ,          0.,          0. ] ))
        bond_in_ref.append(Bond(21, 30, -0.10204218, 1,  [     -1. ,          0.,          0. ] ))
        bond_in_ref.append(Bond(17, 26, -0.10204218, 1,  [     -1. ,          0.,          0. ] ))
        bond_in_ref.append(Bond(20, 31, -0.10204218, 1,  [     -1. ,         -1.,          0. ] ))
        bond_in_ref.append(Bond(16, 27, -0.10204218, 1,  [     -1. ,         -1.,          0. ] ))
        bond_in_ref.append(Bond(22, 29, -0.10204218, 1,  [      0. ,         -1.,          0. ] ))
        bond_in_ref.append(Bond(18, 25, -0.10204218, 1,  [      0. ,         -1.,          0. ] ))
        bond_in_ref.append(Bond( 5, 12, -0.10204198, 1,  [      0. ,          0.,          0. ] ))
        bond_in_ref.append(Bond( 1,  8, -0.10204198, 1,  [     0.  ,        0.  ,       0.    ] ))
        bond_in_ref.append(Bond( 7, 14, -0.10204198, 1,  [      0. ,          0.,          0. ] ))
        bond_in_ref.append(Bond( 3, 10, -0.10204198, 1,  [      0. ,          0.,          0. ] ))
        bond_in_ref.append(Bond( 0,  9, -0.10204198, 1,  [     0.  ,       -1.  ,       0.    ] ))
        bond_in_ref.append(Bond( 4, 13, -0.10204198, 1,  [      0. ,         -1.,          0. ] ))
        bond_in_ref.append(Bond( 2, 11, -0.10204198, 1,  [      0. ,         -1.,          0. ] ))
        bond_in_ref.append(Bond( 6, 15, -0.10204198, 1,  [      0. ,         -1.,          0. ] ))
        bond_in_ref.append(Bond( 5, 13, -0.10202091, 1,  [      0. ,          0.,          0. ] ))
        bond_in_ref.append(Bond( 1,  9, -0.10202091, 1,  [     0.  ,        0.  ,       0.    ] ))
        bond_in_ref.append(Bond( 0,  8, -0.10202091, 1,  [     0.  ,        0.  ,       0.    ] ))
        bond_in_ref.append(Bond( 4, 12, -0.10202091, 1,  [    0.   ,        0.  ,        0.   ] ))
        bond_in_ref.append(Bond( 6, 14, -0.10202091, 1,  [    0.   ,        0.  ,        0.   ] ))
        bond_in_ref.append(Bond( 3, 11, -0.10202091, 1,  [    0.   ,        0.  ,        0.   ] ))
        bond_in_ref.append(Bond( 7, 15, -0.10202091, 1,  [    0.   ,        0.  ,        0.   ] ))
        bond_in_ref.append(Bond( 2, 10, -0.10202091, 1,  [    0.   ,        0.  ,        0.   ] ))
        bond_in_ref.append(Bond(16, 26, -0.10202071, 1,  [   -1.   ,        0.  ,        0.   ] ))
        bond_in_ref.append(Bond(20, 30, -0.10202071, 1,  [   -1.   ,        0.  ,        0.   ] ))
        bond_in_ref.append(Bond(21, 29, -0.10202071, 1,  [    0.   ,        0.  ,        0.   ] ))
        bond_in_ref.append(Bond(17, 25, -0.10202071, 1,  [    0.   ,        0.  ,        0.   ] ))
        bond_in_ref.append(Bond(20, 28, -0.10202071, 1,  [    0.   ,        0.  ,        0.   ] ))
        bond_in_ref.append(Bond(23, 29, -0.10202071, 1,  [    0.   ,        0.  ,        0.   ] ))
        bond_in_ref.append(Bond(23, 31, -0.10202071, 1,  [    0.   ,        0.  ,        0.   ] ))
        bond_in_ref.append(Bond(21, 31, -0.10202071, 1,  [-1,    0.  ,        0.        ]))
        bond_in_ref.append(Bond( 19, 27, -0.10202071, 1,  [    0.   ,        0.  ,        0.        ]))
        bond_in_ref.append(Bond( 19, 25, -0.10202071, 1,  [    0.   ,        0.  ,        0.        ]))
        bond_in_ref.append(Bond( 18, 24, -0.10202071, 1,  [    0.   ,        0.  ,        0.        ]))
        bond_in_ref.append(Bond( 17, 27, -0.10202071, 1,  [   -1.   ,        0.  ,        0.        ]))
        bond_in_ref.append(Bond( 16, 24, -0.10202071, 1,  [    0.   ,        0.  ,        0.        ]))
        bond_in_ref.append(Bond( 22, 28, -0.10202071, 1,  [    0.   ,        0.  ,        0.        ]))
        bond_in_ref.append(Bond( 22, 30, -0.10202071, 1,  [    0.   ,        0.  ,        0.        ]))
        bond_in_ref.append(Bond( 18, 26, -0.10202071, 1,  [    0.   ,        0.  ,        0.        ]))
        bond_in_ref.append(Bond(  4, 15, -0.10202071, 1,  [   -1.   ,       -1.  ,        0.        ]))
        bond_in_ref.append(Bond(  1, 10, -0.10202071, 1,  [   -1.   ,        0.  ,        0.        ]))
        bond_in_ref.append(Bond(  5, 14, -0.10202071, 1,  [   -1.   ,        0.  ,        0.        ]))
        bond_in_ref.append(Bond(  0, 11, -0.10202071, 1,  [   -1.   ,       -1.  ,        0.        ]))
        bond_in_ref.append(Bond(  7, 12, -0.10202071, 1,  [    0.   ,        0.  ,        0.        ]))
        bond_in_ref.append(Bond(  3,  8, -0.10202071, 1,  [    0.   ,   0.       ,  0.        ]))
        bond_in_ref.append(Bond(  2,  9, -0.10202071, 1,  [    0.   ,  -1.       ,  0.        ]))
        bond_in_ref.append(Bond(  6, 13, -0.10202071, 1,  [    0.   ,     -1.     ,      0.        ]))
        bond_in_ref.append(Bond(  0, 20, 1.828 , 4, [  0. ,     0.  ,   -1.   ]))
        bond_in_ref.append(Bond(  2, 22, 1.828 , 4, [  0. ,     0.  ,   -1.   ]))
        bond_in_ref.append(Bond(  1, 21, 1.828 , 4, [  0. ,     0.  ,   -1.   ]))
        bond_in_ref.append(Bond(  3, 23, 1.828 , 4, [  0. ,     0.  ,   -1.   ]))
        bond_in_ref.append(Bond(  2, 18, 1.828 , 4, [  0. ,     0.  ,    0.   ]))
        bond_in_ref.append(Bond(  3, 19, 1.828 , 4, [  0. ,     0.  ,    0.   ]))
        bond_in_ref.append(Bond(  5, 17, 1.828 , 4, [  0. ,     0.  ,    0.   ]))
        bond_in_ref.append(Bond(  4, 16, 1.828 , 4, [  0. ,     0.  ,    0.   ]))
        bond_in_ref.append(Bond(  1, 17, 1.828 , 4, [  0. ,     0.  ,    0.   ]))
        bond_in_ref.append(Bond(  7, 19, 1.828 , 4, [  0. ,     0.  ,    0.   ]))
        bond_in_ref.append(Bond(  6, 18, 1.828 , 4, [  0. ,     0.  ,    0.   ]))
        bond_in_ref.append(Bond(  0, 16, 1.828 , 4, [  0. ,     0.  ,    0.   ]))
        bond_in_ref.append(Bond(  5, 21, 1.828 , 4, [  0. ,     0.  ,    0.   ]))
        bond_in_ref.append(Bond(  4, 20, 1.828 , 4, [  0. ,     0.  ,    0.   ]))
        bond_in_ref.append(Bond(  7, 23, 1.828 , 4, [  0. ,     0.  ,    0.   ]))
        bond_in_ref.append(Bond(  6, 22, 1.828 , 4, [  0. ,     0.  ,    0.   ]))


        bond_in = BondHardness(system, goodBonds)
        assert bond_in == bond_in_ref

    def test_3(self):

        cell = [[ 4.608,  0.   ,  0.   ], [-0.192693,  3.708998,  0.      ],[-2.197042, -1.825517,  3.724799]]
        composition = 4 * ['Mg'] + 8 * ['Al'] + 16 * ['O']
        scaled_positions = [[ 0.5      , 0.5      , 0.5],
        [ 0.765421 , 0.935787 , 0.064529],
        [ 0.797228 , 0.670421 , 0.211417],
        [ 0.459366 , 0.073058 , 0.469905],
        [ 0.435436 , 0.447865 , 0.035142],
        [ 0.985428 , 0.042431 , 0.553334],
        [ 0.994441 , 0.493963 , 0.041761],
        [ 0.239431 , 0.42966  , 0.545479],
        [ 0.7958   , 0.043874 , 0.81675 ],
        [ 0.368644 , 0.935433 , 0.861867],
        [ 0.832236 , 0.510507 , 0.537973],
        [ 0.23725  , 0.932897 , 0.114496],
        [ 0.124872 , 0.111734 , 0.879196],
        [ 0.490994 , 0.851684 , 0.676219],
        [ 0.897435 , 0.22867  , 0.38446 ],
        [ 0.313464 , 0.626413 , 0.237408],
        [ 0.784226 , 0.276025 , 0.67609 ],
        [ 0.714371 , 0.285057 , 0.030802],
        [ 0.577479 , 0.539617 , 0.828825],
        [ 0.172629 , 0.556551 , 0.823111],
        [ 0.719101 , 0.791037 , 0.478798],
        [ 0.510604 , 0.260065 , 0.230066],
        [ 0.090114 , 0.893427 , 0.286456],
        [ 0.081952 , 0.691647 , 0.515724],
        [ 0.117971 , 0.26263  , 0.215085],
        [ 0.313428 , 0.183023 , 0.690758],
        [ 0.903796 , 0.748461 , 0.862665],
        [ 0.527341 , 0.851466 , 0.140968]]
        system = Crystal(symbols=composition, scaled_positions=scaled_positions, cell=cell, pbc=True)
        goodBonds = np.array([[ 0.1 ,  0.14142136,  0.17320508], [ 0.14142136, 0.2 ,  0.24494897],[ 0.17320508,  0.24494897,  0.3]])
        # params = {'symbols': ['Mg', 'Al', 'O'], 'blocks': [[4, 8, 16]], 'fixed': [[1, 1]],
        #           'goodBonds': goodBonds, 'minDistMatrice': np.array([[0.8, 0.8, 0.8], [0.8, 0.8, 0.8], [0.8, 0.8, 0.8]]),
        #           'minAngle': 0, 'minDiagAngle': 0, 'minVectorLength': 0,}
        # config = CrystalConfig(**params)


        bond_in_ref = Bonds()
        bond_in_ref.append(Bond(1 , 2 , -1.44760942668 , 1 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 3 , -1.28681475083 , 2 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 7 , -1.27704990771 , 3 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(3 , 11 , -1.2725385111 , 3 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(1 , 8 , -1.19871107901 , 4 , [ 0.0 , 1.0 , -1.0 ]))
        bond_in_ref.append(Bond(0 , 10 , -1.16728876425 , 4 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(2 , 5 , -1.13848165772 , 5 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(2 , 4 , -1.1317522119 , 5 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(2 , 6 , -1.1204159385 , 5 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(1 , 9 , -1.00182425136 , 6 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(3 , 8 , -0.932096984135 , 7 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(3 , 7 , -0.876229939617 , 8 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(2 , 10 , -0.839602928417 , 8 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(6 , 8 , -0.871359215726 , 9 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(9 , 11 , -0.853940393304 , 9 , [ 0.0 , 0.0 , 1.0 ]))
        bond_in_ref.append(Bond(0 , 21 , -0.788511752891 , 10 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 13 , -0.788210917373 , 10 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(2 , 17 , -0.777837109737 , 10 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(1 , 14 , -0.768012081612 , 10 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(3 , 16 , -0.750445788861 , 10 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(1 , 5 , -0.731018969369 , 11 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 18 , -0.711753122594 , 12 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(1 , 27 , -0.711677042871 , 12 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(2 , 22 , -0.706882085184 , 12 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(1 , 17 , -0.688173022255 , 12 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(1 , 26 , -0.684349357435 , 12 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(2 , 23 , -0.684045194711 , 12 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(1 , 18 , -0.682754389723 , 12 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(2 , 20 , -0.678665090431 , 12 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 15 , -0.673550465609 , 12 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(2 , 27 , -0.665782983111 , 12 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(4 , 9 , -0.700694981581 , 13 , [ 0.0 , -1.0 , -1.0 ]))
        bond_in_ref.append(Bond(6 , 11 , -0.666768437796 , 13 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(7 , 9 , -0.65552343141 , 13 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(10 , 14 , -0.690500850555 , 14 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(5 , 14 , -0.687333027131 , 14 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(9 , 13 , -0.663878550232 , 14 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 8 , -0.676054572893 , 15 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 4 , -0.628525670879 , 15 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(1 , 6 , -0.62616837743 , 15 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(1 , 22 , -0.644078271711 , 16 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(3 , 25 , -0.632453212082 , 16 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(3 , 13 , -0.622512788433 , 16 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(2 , 18 , -0.621589367331 , 16 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(3 , 22 , -0.595536688528 , 16 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 1 , -0.634831032401 , 17 , [ 0.0 , 0.0 , 1.0 ]))
        bond_in_ref.append(Bond(0 , 3 , -0.624466639667 , 17 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(7 , 24 , -0.623353250519 , 18 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(10 , 20 , -0.618594843637 , 18 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(8 , 16 , -0.617585073636 , 18 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(7 , 23 , -0.616917895708 , 18 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(6 , 15 , -0.613556125549 , 18 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(5 , 12 , -0.607112735154 , 18 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(8 , 13 , -0.603835243498 , 18 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(8 , 26 , -0.594728374133 , 18 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(4 , 21 , -0.59064128896 , 18 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(5 , 8 , -0.603547105599 , 19 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(5 , 7 , -0.591345955964 , 19 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(5 , 9 , -0.556799902176 , 19 , [ 1.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 11 , -0.598702007691 , 20 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(8 , 17 , -0.571876980219 , 21 , [ 0.0 , 0.0 , 1.0 ]))
        bond_in_ref.append(Bond(7 , 25 , -0.570000792626 , 21 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(11 , 22 , -0.558180344557 , 21 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(10 , 26 , -0.554277228491 , 21 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(5 , 25 , -0.553443968697 , 21 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(4 , 15 , -0.552121825672 , 21 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(5 , 20 , -0.550321222726 , 21 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(10 , 16 , -0.545960000984 , 21 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(9 , 27 , -0.538564250198 , 21 , [ 0.0 , 0.0 , 1.0 ]))
        bond_in_ref.append(Bond(11 , 27 , -0.525527868302 , 21 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 20 , -0.567749771014 , 22 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(3 , 15 , -0.562048063872 , 22 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(1 , 13 , -0.518949743762 , 22 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(1 , 11 , -0.544159934948 , 23 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(3 , 10 , -0.495788350296 , 23 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(9 , 12 , -0.520840011966 , 24 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(11 , 26 , -0.517811819363 , 24 , [ -1.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(4 , 25 , -0.516941603883 , 24 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(4 , 19 , -0.515354631641 , 24 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(6 , 24 , -0.509725417791 , 24 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(4 , 27 , -0.503248692849 , 24 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(10 , 23 , -0.502181918244 , 24 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(8 , 20 , -0.49916925227 , 24 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(6 , 16 , -0.495153194502 , 24 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(5 , 23 , -0.494330404329 , 24 , [ 1.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(11 , 24 , -0.485630801483 , 24 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(9 , 25 , -0.482119548167 , 24 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(6 , 22 , -0.48005325087 , 24 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(8 , 12 , -0.477424147227 , 24 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(7 , 10 , -0.518319860037 , 25 , [ -1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(5 , 10 , -0.49733775904 , 25 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 24 , -0.507538803586 , 26 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(7 , 19 , -0.470520168897 , 27 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(7 , 18 , -0.46954330625 , 27 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(11 , 12 , -0.468669797124 , 27 , [ 0.0 , 1.0 , -1.0 ]))
        bond_in_ref.append(Bond(7 , 14 , -0.468542369508 , 27 , [ -1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(9 , 23 , -0.455688356409 , 27 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(10 , 21 , -0.447515546401 , 27 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(6 , 26 , -0.432812560114 , 27 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(6 , 17 , -0.42948973065 , 27 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(11 , 21 , -0.428159532259 , 27 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(11 , 15 , -0.427112198422 , 27 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(11 , 19 , -0.422055687698 , 27 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(3 , 21 , -0.455450686828 , 28 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(3 , 20 , -0.443203175746 , 28 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 16 , -0.429614440007 , 28 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(1 , 24 , -0.424373270747 , 28 , [ 1.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(3 , 27 , -0.422696874422 , 28 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(4 , 17 , -0.415934138808 , 29 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(10 , 19 , -0.41356600012 , 29 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(5 , 16 , -0.399006842476 , 29 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(5 , 22 , -0.387824144586 , 29 , [ 1.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(9 , 21 , -0.383027255037 , 29 , [ 0.0 , 1.0 , 1.0 ]))
        bond_in_ref.append(Bond(8 , 10 , -0.415706385504 , 30 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(4 , 7 , -0.409900805254 , 30 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(8 , 11 , -0.393601404544 , 30 , [ 1.0 , -1.0 , 1.0 ]))
        bond_in_ref.append(Bond(5 , 10 , -0.392617038721 , 30 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(4 , 11 , -0.366136736129 , 30 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 9 , -0.399492470577 , 31 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(1 , 4 , -0.394978433802 , 31 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 26 , -0.359897897504 , 32 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(3 , 18 , -0.322095275851 , 32 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(4 , 6 , -0.358190458814 , 33 , [ -1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(8 , 9 , -0.310149580351 , 33 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(4 , 18 , -0.355737328445 , 34 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(4 , 12 , -0.349854531446 , 34 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(9 , 19 , -0.334768513311 , 34 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(1 , 7 , -0.342974995052 , 35 , [ 1.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(3 , 9 , -0.338816886278 , 35 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(3 , 5 , -0.336379275084 , 35 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 6 , -0.312840415727 , 35 , [ -1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(4 , 10 , -0.302362976179 , 36 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(4 , 5 , -0.301450527911 , 36 , [ -1.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(6 , 10 , -0.272767993165 , 36 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(0 , 2 , -0.293615837613 , 37 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(1 , 2 , -0.283226454628 , 37 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(1 , 10 , -0.290455789647 , 38 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(1 , 6 , -0.282772525587 , 38 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(3 , 6 , -0.253918442024 , 38 , [ -1.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(2 , 7 , -0.247323841492 , 38 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(8 , 24 , -0.279313583623 , 39 , [ 1.0 , 0.0 , 1.0 ]))
        bond_in_ref.append(Bond(7 , 13 , -0.250686313779 , 39 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(6 , 12 , -0.230543507282 , 39 , [ 1.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(7 , 11 , -0.238760129419 , 40 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(6 , 7 , -0.21487523995 , 40 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(4 , 9 , -0.199330142219 , 40 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(3 , 24 , -0.234801162724 , 41 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(2 , 14 , -0.200528677719 , 41 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(2 , 8 , -0.233626978095 , 42 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(3 , 5 , -0.223540904224 , 42 , [ -1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(3 , 4 , -0.195003704025 , 42 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(2 , 9 , -0.189865167135 , 42 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(9 , 17 , -0.225046286763 , 43 , [ 0.0 , 1.0 , 1.0 ]))
        bond_in_ref.append(Bond(6 , 19 , -0.219423428713 , 43 , [ 1.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(2 , 3 , -0.208316970367 , 44 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(1 , 7 , -0.168784853396 , 45 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(3 , 6 , -0.154785994033 , 45 , [ 0.0 , 0.0 , 1.0 ]))
        bond_in_ref.append(Bond(1 , 4 , -0.121400853441 , 45 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(2 , 11 , -0.120019714832 , 45 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(10 , 11 , -0.1518599357 , 46 , [ 1.0 , 0.0 , 1.0 ]))
        bond_in_ref.append(Bond(5 , 6 , -0.145298256865 , 46 , [ 0.0 , 0.0 , 1.0 ]))
        bond_in_ref.append(Bond(9 , 10 , -0.130480096409 , 46 , [ -1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(4 , 11 , -0.120710628739 , 46 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(4 , 20 , -0.133321861074 , 47 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(5 , 19 , -0.12660288173 , 47 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(6 , 18 , -0.11532023857 , 47 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(1 , 21 , -0.132892522168 , 48 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 6 , -0.113698522026 , 49 , [ 0.0 , 0.0 , 1.0 ]))
        bond_in_ref.append(Bond(3 , 9 , -0.0940037207948 , 49 , [ 0.0 , -1.0 , -1.0 ]))
        bond_in_ref.append(Bond(1 , 11 , -0.0681270382781 , 49 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(1 , 3 , -0.0877064243356 , 50 , [ 0.0 , 1.0 , -1.0 ]))
        bond_in_ref.append(Bond(0 , 2 , -0.0782044877177 , 50 , [ 0.0 , 0.0 , 1.0 ]))
        bond_in_ref.append(Bond(1 , 3 , -0.0751024496108 , 50 , [ 1.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(7 , 22 , -0.072928457092 , 51 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(9 , 18 , -0.072319178444 , 51 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(3 , 4 , -0.05293577234 , 52 , [ 0.0 , 0.0 , 1.0 ]))
        bond_in_ref.append(Bond(1 , 10 , -0.0473278299127 , 52 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(2 , 21 , -0.052240520432 , 53 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(2 , 26 , -0.0480369959256 , 53 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(2 , 14 , -0.00536874742255 , 53 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(1 , 3 , -0.0361287600925 , 54 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(2 , 3 , -0.034756709884 , 54 , [ 1.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 25 , 0.00606319748182 , 55 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(3 , 17 , 0.0349857076056 , 55 , [ 0.0 , 0.0 , 1.0 ]))
        bond_in_ref.append(Bond(0 , 23 , 0.0418892912567 , 55 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(6 , 23 , 0.0250766694094 , 56 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(5 , 27 , 0.0601893113107 , 56 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 8 , 0.0473912427826 , 57 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(2 , 8 , 0.0532279132098 , 57 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(2 , 9 , 0.0613574630477 , 57 , [ 0.0 , -1.0 , -1.0 ]))
        bond_in_ref.append(Bond(2 , 8 , 0.0663488404393 , 57 , [ 0.0 , 1.0 , -1.0 ]))
        bond_in_ref.append(Bond(2 , 9 , 0.0709199513809 , 57 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(2 , 11 , 0.0819764261017 , 57 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 9 , 0.0879175597571 , 57 , [ 0.0 , -1.0 , -1.0 ]))
        bond_in_ref.append(Bond(2 , 7 , 0.0970437898162 , 57 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(6 , 11 , 0.0524578677246 , 58 , [ 1.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(2 , 3 , 0.0794242612458 , 59 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(2 , 24 , 0.099330306626 , 60 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(5 , 11 , 0.113469100732 , 61 , [ 1.0 , -1.0 , 1.0 ]))
        bond_in_ref.append(Bond(5 , 6 , 0.121191819511 , 61 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(4 , 6 , 0.137529790686 , 61 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(8 , 10 , 0.1434035437 , 61 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 4 , 0.115639234696 , 62 , [ 0.0 , 0.0 , 1.0 ]))
        bond_in_ref.append(Bond(3 , 7 , 0.130559185437 , 62 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 11 , 0.13590335867 , 62 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(2 , 7 , 0.164811605116 , 62 , [ 1.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(21 , 27 , 0.116646443604 , 63 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(18 , 25 , 0.137425656893 , 63 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(12 , 24 , 0.165363261895 , 63 , [ 0.0 , 0.0 , 1.0 ]))
        bond_in_ref.append(Bond(7 , 15 , 0.116693423 , 64 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(10 , 17 , 0.118593465378 , 64 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(3 , 10 , 0.16643001788 , 65 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 5 , 0.166694517475 , 65 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(3 , 4 , 0.206416876926 , 65 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(15 , 22 , 0.182385896439 , 66 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(13 , 25 , 0.191325969153 , 66 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(19 , 26 , 0.192491775834 , 66 , [ -1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(20 , 27 , 0.201146430315 , 66 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(15 , 24 , 0.206490752131 , 66 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(22 , 23 , 0.210836016509 , 66 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(17 , 18 , 0.211461430422 , 66 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(22 , 24 , 0.21800734493 , 66 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(22 , 25 , 0.225320230858 , 66 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(13 , 18 , 0.229368597314 , 66 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(4 , 8 , 0.186073471321 , 67 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(5 , 7 , 0.188909844539 , 67 , [ 1.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(8 , 9 , 0.192280317845 , 67 , [ 1.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(4 , 5 , 0.216861636551 , 67 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(6 , 8 , 0.222816410262 , 67 , [ 0.0 , 1.0 , -1.0 ]))
        bond_in_ref.append(Bond(7 , 11 , 0.229390612447 , 67 , [ 0.0 , 0.0 , 1.0 ]))
        bond_in_ref.append(Bond(2 , 15 , 0.219577192749 , 68 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(1 , 20 , 0.22212738943 , 68 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(3 , 14 , 0.248480345542 , 68 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 14 , 0.250967530723 , 68 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(3 , 23 , 0.252458852453 , 68 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 5 , 0.232055774106 , 69 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(2 , 10 , 0.268629543682 , 69 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(10 , 12 , 0.235045025459 , 70 , [ 1.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(8 , 18 , 0.241370901712 , 70 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(8 , 18 , 0.268892750487 , 70 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(19 , 27 , 0.241651157909 , 71 , [ 0.0 , 0.0 , 1.0 ]))
        bond_in_ref.append(Bond(18 , 26 , 0.24638033115 , 71 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(14 , 17 , 0.252221593662 , 71 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(12 , 25 , 0.252742511619 , 71 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(16 , 26 , 0.25425288051 , 71 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(15 , 27 , 0.255614623616 , 71 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(19 , 23 , 0.258857890186 , 71 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(14 , 24 , 0.260119566376 , 71 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(15 , 20 , 0.275570166731 , 71 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(22 , 26 , 0.279233307838 , 71 , [ -1.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(12 , 26 , 0.283521726679 , 71 , [ -1.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(19 , 25 , 0.284549888798 , 71 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(14 , 21 , 0.289346696145 , 71 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(1 , 3 , 0.2426793769 , 72 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(0 , 2 , 0.254611289485 , 72 , [ -1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 1 , 0.270698776612 , 72 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(7 , 9 , 0.263302839812 , 73 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(10 , 11 , 0.273145261404 , 73 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(5 , 11 , 0.306090484619 , 73 , [ 1.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(9 , 10 , 0.30789073506 , 73 , [ 0.0 , 1.0 , 1.0 ]))
        bond_in_ref.append(Bond(2 , 25 , 0.270755370995 , 74 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(2 , 15 , 0.271698328924 , 74 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 12 , 0.280570852706 , 74 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(1 , 17 , 0.285610955864 , 74 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(1 , 12 , 0.291751061511 , 74 , [ 1.0 , 1.0 , -1.0 ]))
        bond_in_ref.append(Bond(2 , 16 , 0.30342378783 , 74 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(10 , 18 , 0.288633489304 , 75 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(8 , 25 , 0.290727652311 , 75 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(4 , 24 , 0.321377494891 , 75 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 1 , 0.29368100953 , 76 , [ -1.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(16 , 18 , 0.294027297166 , 77 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(14 , 22 , 0.301774029664 , 77 , [ 1.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(17 , 21 , 0.305251377272 , 77 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(14 , 23 , 0.306537426315 , 77 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(18 , 27 , 0.308046156666 , 77 , [ 0.0 , 0.0 , 1.0 ]))
        bond_in_ref.append(Bond(16 , 20 , 0.311326353238 , 77 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(12 , 21 , 0.31483294832 , 77 , [ 0.0 , 0.0 , 1.0 ]))
        bond_in_ref.append(Bond(14 , 16 , 0.316971734955 , 77 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(13 , 17 , 0.321327223705 , 77 , [ 0.0 , 1.0 , 1.0 ]))
        bond_in_ref.append(Bond(13 , 23 , 0.323768603741 , 77 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(15 , 19 , 0.324400021067 , 77 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(13 , 15 , 0.325785024091 , 77 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(7 , 8 , 0.321163192116 , 78 , [ -1.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(6 , 10 , 0.321891661091 , 78 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(7 , 10 , 0.330631911597 , 78 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(4 , 8 , 0.333573662261 , 78 , [ -1.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(4 , 10 , 0.339796468285 , 78 , [ -1.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(7 , 8 , 0.340436880536 , 78 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(2 , 25 , 0.327039892638 , 79 , [ 1.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 22 , 0.336061939478 , 79 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 27 , 0.365572157337 , 79 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(17 , 24 , 0.346309578072 , 80 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(20 , 26 , 0.347924966677 , 80 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(13 , 20 , 0.352944006614 , 80 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(15 , 21 , 0.353651885668 , 80 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(20 , 23 , 0.35370452713 , 80 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(17 , 25 , 0.362585314391 , 80 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(12 , 16 , 0.37330887918 , 80 , [ -1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(16 , 19 , 0.380501704436 , 80 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(14 , 19 , 0.38301641981 , 80 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(14 , 27 , 0.394289153501 , 80 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(9 , 26 , 0.34935388158 , 81 , [ -1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(10 , 13 , 0.379996833399 , 81 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(5 , 26 , 0.380536048774 , 81 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 5 , 0.383395035736 , 82 , [ -1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(1 , 23 , 0.395571280042 , 83 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(2 , 19 , 0.417248633785 , 83 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(1 , 25 , 0.42202916261 , 83 , [ 0.0 , 1.0 , -1.0 ]))
        bond_in_ref.append(Bond(1 , 16 , 0.430559785735 , 83 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(1 , 20 , 0.431259200904 , 83 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(1 , 19 , 0.434068234695 , 83 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(0 , 22 , 0.437563645637 , 83 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(7 , 27 , 0.399743318837 , 84 , [ 0.0 , 0.0 , 1.0 ]))
        bond_in_ref.append(Bond(8 , 22 , 0.407996453678 , 84 , [ 1.0 , -1.0 , 1.0 ]))
        bond_in_ref.append(Bond(8 , 14 , 0.423083356602 , 84 , [ 0.0 , 0.0 , 1.0 ]))
        bond_in_ref.append(Bond(10 , 15 , 0.428302297895 , 84 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(7 , 21 , 0.428826416963 , 84 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(10 , 27 , 0.434024018944 , 84 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(9 , 14 , 0.437544555055 , 84 , [ 0.0 , 1.0 , 1.0 ]))
        bond_in_ref.append(Bond(11 , 20 , 0.444087305487 , 84 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(5 , 24 , 0.446086905466 , 84 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(23 , 25 , 0.420420266142 , 85 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(12 , 23 , 0.440105059816 , 85 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(21 , 24 , 0.45809136518 , 85 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(23 , 24 , 0.459654635681 , 85 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(12 , 19 , 0.463317969095 , 85 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 11 , 0.434405571665 , 86 , [ 1.0 , 0.0 , 1.0 ]))
        bond_in_ref.append(Bond(0 , 8 , 0.448639198715 , 86 , [ -1.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(2 , 10 , 0.460868939359 , 86 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(1 , 10 , 0.461305233393 , 86 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 27 , 0.448662804763 , 87 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(2 , 12 , 0.452541615187 , 87 , [ 1.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(2 , 16 , 0.456643123891 , 87 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(3 , 19 , 0.456669435315 , 87 , [ 0.0 , -1.0 , -1.0 ]))
        bond_in_ref.append(Bond(3 , 12 , 0.459226168915 , 87 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(3 , 26 , 0.469728322408 , 87 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(3 , 26 , 0.475294298221 , 87 , [ -1.0 , -1.0 , -1.0 ]))
        bond_in_ref.append(Bond(3 , 14 , 0.489620083368 , 87 , [ -1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 19 , 0.491790419578 , 87 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(9 , 22 , 0.452534687616 , 88 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(7 , 17 , 0.459692772761 , 88 , [ -1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(11 , 23 , 0.460538255189 , 88 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(5 , 13 , 0.46939201047 , 88 , [ 1.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(6 , 14 , 0.483699496437 , 88 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(11 , 25 , 0.485963682206 , 88 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(4 , 13 , 0.48645327044 , 88 , [ 0.0 , -1.0 , -1.0 ]))
        bond_in_ref.append(Bond(6 , 9 , 0.486553849886 , 90 , [ 1.0 , -1.0 , -1.0 ]))
        bond_in_ref.append(Bond(6 , 9 , 0.487330470094 , 90 , [ 1.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(8 , 11 , 0.515789812928 , 90 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(4 , 7 , 0.528869497205 , 90 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(3 , 6 , 0.490822553044 , 91 , [ -1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(3 , 8 , 0.495940579228 , 91 , [ -1.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(1 , 8 , 0.500961538621 , 91 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(1 , 5 , 0.521673590206 , 91 , [ 0.0 , 1.0 , -1.0 ]))
        bond_in_ref.append(Bond(2 , 24 , 0.505589059136 , 92 , [ 1.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(1 , 25 , 0.519570627822 , 92 , [ 1.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(1 , 16 , 0.526762600479 , 92 , [ 0.0 , 1.0 , -1.0 ]))
        bond_in_ref.append(Bond(2 , 13 , 0.531406969257 , 92 , [ 0.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(2 , 21 , 0.540875485703 , 92 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(1 , 23 , 0.546788631163 , 92 , [ 1.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(8 , 15 , 0.507655391348 , 93 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(5 , 21 , 0.51995545524 , 93 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(11 , 15 , 0.522667273668 , 93 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(4 , 14 , 0.531604366357 , 93 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(7 , 22 , 0.534857499249 , 93 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(10 , 24 , 0.549937623155 , 93 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 10 , 0.54581627494 , 95 , [ -1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(6 , 7 , 0.553709995598 , 96 , [ 1.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(5 , 11 , 0.56176858034 , 96 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(7 , 8 , 0.563442962229 , 96 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(5 , 9 , 0.593079718433 , 96 , [ 0.0 , -1.0 , -1.0 ]))
        bond_in_ref.append(Bond(4 , 8 , 0.599859483519 , 96 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 14 , 0.560828986611 , 97 , [ -1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(3 , 15 , 0.567517495066 , 97 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(1 , 16 , 0.568046844737 , 97 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 17 , 0.584676015003 , 97 , [ 0.0 , 0.0 , 1.0 ]))
        bond_in_ref.append(Bond(3 , 13 , 0.595806341869 , 97 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(3 , 9 , 0.602734053011 , 99 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(2 , 5 , 0.605007731428 , 99 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 5 , 0.609192565302 , 99 , [ -1.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 19 , 0.622787420684 , 102 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 25 , 0.626082569015 , 102 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(2 , 17 , 0.626712883769 , 102 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 23 , 0.628451826295 , 102 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 17 , 0.632188206871 , 102 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(3 , 12 , 0.641849198036 , 102 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(3 , 26 , 0.647828916852 , 102 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(1 , 15 , 0.65360017884 , 102 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(2 , 3 , 0.671189050107 , 106 , [ 0.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 2 , 0.692718495389 , 106 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(3 , 8 , 0.671967709542 , 107 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 9 , 0.707539982977 , 107 , [ 0.0 , -1.0 , 0.0 ]))
        bond_in_ref.append(Bond(1 , 9 , 0.712598720789 , 107 , [ 1.0 , 0.0 , -1.0 ]))
        bond_in_ref.append(Bond(1 , 8 , 0.718784077469 , 107 , [ 0.0 , 1.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 7 , 0.722987831239 , 111 , [ 1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(2 , 6 , 0.750275712322 , 111 , [ -1.0 , 0.0 , 0.0 ]))
        bond_in_ref.append(Bond(0 , 1 , 0.762998361337 , 115 , [ 0.0 , 0.0 , 0.0 ]))

        bond_in = BondHardness(system, goodBonds)
        assert bond_in == bond_in_ref

    def test_MgAlO_new1(self):
        symbols = 4*['Mg'] + 8*['Al'] + 16*['O']
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

        goodBonds = [[0.100000000000000 ,  0.141421356237310 ,  0.173205080756888],
                     [0.141421356237310 ,  0.200000000000000 ,  0.244948974278318],
                     [0.173205080756888 ,  0.244948974278318 ,  0.300000000000000]]

        system = AtomicStructure(symbols=symbols, scaled_positions=scaled_positions, cell=cell, pbc=True)

        bond_in = BondHardness_new(system, goodBonds)

        print('1')


    def test_MgO_new1(self):
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

        goodBonds = [[0.1000 ,   0.1732],
                     [0.1732 ,   0.3000]]

        system = AtomicStructure(symbols=symbols, scaled_positions=scaled_positions, cell=cell, pbc=True)
        bond_in = BondHardness_new(system, goodBonds)
        print('MgO_new1')
