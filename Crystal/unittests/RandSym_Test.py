import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


'''
@file        RandomTest.py
@author:     Evgeny Tikhonov
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    e.tikhonov@physics.msu.ru
@date        1 December 2016
@brief       Class for testing random for Crystal structures.
'''


import unittest
import os
import json

from ...Atomistic.mol.read_molecule import read_molecule
from ...SystemPool import SystemPool
from ...Atomistic.Crystal import Crystal
from ...Atomistic.CompositionSpace import CompositionSpace
from ..RandSym import RandSym, VOFailed


HOMEPATH = os.path.dirname(os.path.abspath(__file__))


# class RandSym_Test(unittest.TestCase):
#     def test_atomic_fixed(self):
#         config = {'externalPressure' : 100}
#         compositionSpace = CompositionSpace(symbols = ['Mg', 'Al', 'O'], blocks = [[4, 8, 16]], range = [[1, 1]])
#         pool = SystemPool()
#         randtop = RandSym(Crystal, config, pool, {'compositionSpace' : compositionSpace})
#         randtop.prepare()
#         count = 0
#         while count < 1:
#             try:
#                 offsprings = randtop()
#                 count += len(offsprings)
#             except VOFailed:
#                 pass
#
#         randtop.standby()
#
#     def test_atomic_variable(self):
#         config = {}
#         compositionSpace = CompositionSpace(symbols = ['Mo', 'B'], blocks = [[1, 0], [0, 1]], range = [[0, 18], [0, 18]])
#         pool = SystemPool()
#         randtop = RandSym(Crystal, config, pool, {'compositionSpace' : compositionSpace})
#         randtop.prepare()
#         count = 0
#         while count < 1:
#             try:
#                 offsprings = randtop()
#                 count += len(offsprings)
#             except VOFailed:
#                 pass
#         randtop.standby()
#     #
#     # def test_molecular_fixed(self):
#     #     mol = read_molecule(f'{HOMEPATH}/MOL_glycine')
#     #     params = {'symbols' : [mol], 'blocks' : [[4]], 'fixed' : [[1, 1]]}
#     #     config = CrystalConfig(**params)
#     #     pool = CrystalPool(config)
#     #     randtop = RandSym(config, pool, initFrac=1.0)
#     #     randtop.prepare()
#     #     count = 0
#     #     while count < 10:
#     #         try:
#     #             offsprings = randtop()
#     #             count += len(offsprings)
#     #         except VOFailed:
#     #             pass
#     #     randtop.standby()
#     #
#     # def test_molecular_variable(self):
#     #     mol_1 = read_molecule(f'{HOMEPATH}/MOL_1')
#     #     mol_2 = read_molecule(f'{HOMEPATH}/MOL_2')
#     #     mol_3 = read_molecule(f'{HOMEPATH}/MOL_3')
#     #     mol_5 = read_molecule(f'{HOMEPATH}/MOL_5')
#     #     params = {'symbols': [mol_1, mol_2, mol_3, 'O', mol_5],
#     #               'blocks': [[0,2,0,1,0], [1,0,0,0,0],[0,0,1,0,0],[0,0,0,0,1]],
#     #               'fixed': [[1,2],[0,4],[0,4],[0,4]], 'minAt': 5, 'maxAt': 40,
#     #               'volumeType' : 'atom', 'externalPressure' : 200,
#     #               'ionDistances' : {'C-C' : 1.20, 'C-O' : 1.20, 'C-H' : 1.20, 'C-N' : 1.20,
#     #                                 'O-C' : 1.20, 'O-O' : 1.20, 'O-H' : 1.20, 'O-N' : 1.20,
#     #                                 'H-C' : 1.20, 'H-O' : 1.20, 'H-H' : 0.51, 'H-N' : 1.20,
#     #                                 'N-C' : 1.20, 'N-O' : 1.20, 'N-H' : 1.20, 'N-N' : 1.20}}
#     #     config = CrystalConfig(**params)
#     #     pool = CrystalPool(config)
#     #     randtop = RandSym(config, pool, initFrac=1.0)
#     #     randtop.prepare()
#     #     count = 0
#     #     while count < 10:
#     #         try:
#     #             offsprings = randtop()
#     #             count += len(offsprings)
#     #         except VOFailed:
#     #             pass
#     #     randtop.standby()
