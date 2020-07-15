import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


'''
@file        SeedsTest.py
@author:     Pavel Bushlanov
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    paulbusj@mail.ru
@date        12 Septenber 2017
@brief       Class for testing seeds for Crystal structures.
'''


import os
import unittest
import toml

from ...Atomistic.mol.read_molecule import read_molecule
from ...Atomistic.Crystal import Crystal
from ...Atomistic.CrystalPool import CrystalPool
from ..Seeds import Seeds, VOFailed


HOMEPATH = os.path.dirname(os.path.abspath(__file__))


class Seeds_Test(unittest.TestCase):
    def test_atomic_fixed(self):
        config = {'externalPressure' : 100}
        pool = CrystalPool(symbols = ['Mg', 'Al', 'O'], blocks = [[4, 8, 16]], range = [[1, 1]])
        seeds_folder = os.path.join(HOMEPATH, 'Seeds/MgAl2O4')
        seeds = Seeds(Crystal, config, pool, generations=[0], seedsFolders=[seeds_folder])
        seeds.prepare()
        count = 0
        while count < 1:
            try:
                offsprings = seeds()
                count += len(offsprings)
            except VOFailed:
                pass

        seeds.standby()
        # population = []
        # population.append(offsprings[0].toDICT())
        # with open(f'{HOMEPATH}/atomic_structures_fixed', 'wt') as f:
        #     f.write(json.dumps(population))


    def test_molecular_variable(self):
        mol_1 = read_molecule(f'{HOMEPATH}/MOL_1')
        mol_2 = read_molecule(f'{HOMEPATH}/MOL_2')
        mol_3 = read_molecule(f'{HOMEPATH}/MOL_3')
        mol_5 = read_molecule(f'{HOMEPATH}/MOL_5')
        config = {'volumeType' : 0, 'externalPressure' : 200,
                  'ionDistances' : {('C','C') : 1.20, ('C','O') : 1.20, ('C','H') : 1.20, ('C','N') : 1.20,
                                    ('O','C') : 1.20, ('O','O') : 1.20, ('O','H') : 1.20, ('O','N') : 1.20,
                                    ('H','C') : 1.20, ('H','O') : 1.20, ('H','H') : 0.51, ('H','N') : 1.20,
                                    ('N','C') : 1.20, ('N','O') : 1.20, ('N','H') : 1.20, ('N','N') : 1.20}}
        pool = CrystalPool(symbols = [mol_1, mol_2, mol_3, 'O', mol_5],
                           blocks = [[0,2,0,1,0], [1,0,0,0,0],[0,0,1,0,0],[0,0,0,0,1]],
                           range = [[1,2],[0,4],[0,4],[0,4]], minAt = 5, maxAt = 40)
        seeds_folder = os.path.join(HOMEPATH, 'Seeds/CNHO')
        seeds = Seeds(Crystal, config, pool, generations=[0], seedsFolders=[seeds_folder])
        seeds.prepare()
        count = 0
        while count < 1:
            try:
                offsprings = seeds()
                count += len(offsprings)
            except VOFailed:
                pass

        seeds.standby()
        # population = []
        # population.append(offsprings[0].toDICT())
        # with open(f'{HOMEPATH}/molecular_structures_variable', 'wt') as f:
        #     f.write(json.dumps(population))
