import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import unittest
import os
import json

from ..mol.read_molecule import read_molecule
from ...SystemPool import SystemPool
from ..Crystal import Crystal
from ...Atomistic.CompositionSpace import CompositionSpace
from ..Rotation import Rotation, VOFailed


HOMEPATH = os.path.dirname(os.path.abspath(__file__))

#
# class Rotation_Test(unittest.TestCase):
#
#     def test_molecular_fixed(self):
#         mol_glycine = read_molecule(f'{HOMEPATH}/MOL_glycine')
#         mol_H2O = read_molecule(f'{HOMEPATH}/MOL_H2O')
#         config = {}
#         compositionSpace = CompositionSpace(symbols = [mol_glycine, mol_H2O], blocks = [[4, 2]], range = [[1, 1]])
#         pool = SystemPool()
#         rotation = Rotation(Crystal, config, pool, {'compositionSpace' : compositionSpace})
#
#         with open(f'{HOMEPATH}/molecular_structures_fixed', 'rt') as f:
#             population = json.loads(f.read())
#         population_iterator = iter(population)
#         count = 0
#         while count < 1:
#             try:
#                 parent = {'structure': Crystal.fromDICT(next(population_iterator)), 'ID': 0}
#                 offsprings = rotation(parent)
#                 count += len(offsprings)
#             except VOFailed:
#                 pass
#
#     def test_molecular_variable(self):
#         mol_1 = read_molecule(f'{HOMEPATH}/MOL_1')
#         mol_2 = read_molecule(f'{HOMEPATH}/MOL_2')
#         mol_3 = read_molecule(f'{HOMEPATH}/MOL_3')
#         mol_5 = read_molecule(f'{HOMEPATH}/MOL_5')
#         config = {'volumeType' : 0, 'externalPressure' : 200,
#                   'ionDistances' : {('C','C') : 1.20, ('C','O') : 1.20, ('C','H') : 1.20, ('C','N') : 1.20,
#                                     ('O','C') : 1.20, ('O','O') : 1.20, ('O','H') : 1.20, ('O','N') : 1.20,
#                                     ('H','C') : 1.20, ('H','O') : 1.20, ('H','H') : 0.51, ('H','N') : 1.20,
#                                     ('N','C') : 1.20, ('N','O') : 1.20, ('N','H') : 1.20, ('N','N') : 1.20}}
#         compositionSpace = CompositionSpace(symbols = [mol_1, mol_2, mol_3, 'O', mol_5],
#                            blocks = [[0,2,0,1,0], [1,0,0,0,0],[0,0,1,0,0],[0,0,0,0,1]],
#                            range = [[1,2],[0,4],[0,4],[0,4]], minAt = 5, maxAt = 40)
#         pool = SystemPool()
#         rotation = Rotation(Crystal, config, pool, {'compositionSpace' : compositionSpace})
#
#         with open(f'{HOMEPATH}/molecular_structures_variable', 'rt') as f:
#             population = json.loads(f.read())
#         population_iterator = iter(population)
#         count = 0
#         while count < 1:
#             try:
#                 parent = {'structure': Crystal.fromDICT(next(population_iterator)), 'ID': 0}
#                 parent['structure'].config = config
#                 offsprings = rotation(parent)
#                 count += len(offsprings)
#             except VOFailed:
#                 pass
