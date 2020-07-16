import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


'''
@file        TwinningTest.py
@author:     Pavel Bushlanov
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    paulbush@mail.ru
@date        31 October 2018
@brief       Class for testing twinning for Crystal structures.
'''


import unittest
import os
import json

from ...Atomistic.mol.read_molecule import read_molecule
from ...SystemPool import SystemPool
from ...Atomistic.Crystal import Crystal
from ...Atomistic.CompositionSpace import CompositionSpace
from ..Twinning import Twinning, VOFailed


HOMEPATH = os.path.dirname(os.path.abspath(__file__))


class Twinning_Test(unittest.TestCase):
    def test_atomic_fixed(self):
        config = {'externalPressure' : 100}
        compositionSpace = CompositionSpace(symbols = ['Mg', 'Al', 'O'], blocks = [[4, 8, 16]], range = [[1, 1]])
        pool = SystemPool()
        twinning = Twinning(Crystal, config, pool, {'compositionSpace' : compositionSpace})
        twinning.correlation_coefficient = 0.41

        with open(f'{HOMEPATH}/atomic_structures_fixed', 'rt') as f:
            population = json.loads(f.read())
        population_iterator = iter(population)
        count = 0
        while count < 1:
            try:
                parent = Crystal.fromDICT(next(population_iterator))
                offsprings = twinning(parent)
                count += len(offsprings)
            except VOFailed:
                pass

    def test_atomic_variable(self):
        config = {}
        compositionSpace = CompositionSpace(symbols = ['Mo', 'B'], blocks = [[1, 0], [0, 1]], range = [[0, 18], [0, 18]])
        pool = SystemPool()
        twinning = Twinning(Crystal, config, pool, {'compositionSpace' : compositionSpace})
        twinning.correlation_coefficient = 0.41

        with open(f'{HOMEPATH}/atomic_structures_variable', 'rt') as f:
            population = json.loads(f.read())
        population_iterator = iter(population)
        count = 0
        while count < 1:
            try:
                parent = Crystal.fromDICT(next(population_iterator))
                offsprings = twinning(parent)
                count += len(offsprings)
            except VOFailed:
                pass

    def test_molecular_fixed(self):
        mol_glycine = read_molecule(f'{HOMEPATH}/MOL_glycine')
        mol_H2O = read_molecule(f'{HOMEPATH}/MOL_H2O')
        config = {}
        compositionSpace = CompositionSpace(symbols = [mol_glycine, mol_H2O], blocks = [[4, 2]], range = [[1, 1]])
        pool = SystemPool()
        twinning = Twinning(Crystal, config, pool, {'compositionSpace' : compositionSpace})
        twinning.correlation_coefficient = 0.41

        with open(f'{HOMEPATH}/molecular_structures_fixed', 'rt') as f:
            population = json.loads(f.read())
        population_iterator = iter(population)
        count = 0
        while count < 1:
            try:
                parent = Crystal.fromDICT(next(population_iterator))
                offsprings = twinning(parent)
                count += len(offsprings)
            except VOFailed:
                pass

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
        compositionSpace = CompositionSpace(symbols = [mol_1, mol_2, mol_3, 'O', mol_5],
                           blocks = [[0,2,0,1,0], [1,0,0,0,0],[0,0,1,0,0],[0,0,0,0,1]],
                           range = [[1,2],[0,4],[0,4],[0,4]], minAt = 5, maxAt = 40)
        pool = SystemPool()
        twinning = Twinning(Crystal, config, pool, {'compositionSpace' : compositionSpace})
        twinning.correlation_coefficient = 0.41

        with open(f'{HOMEPATH}/molecular_structures_variable', 'rt') as f:
            population = json.loads(f.read())
        population_iterator = iter(population)
        count = 0
        while count < 1:
            try:
                parent = Crystal.fromDICT(next(population_iterator))
                offsprings = twinning(parent)
                count += len(offsprings)
            except VOFailed:
                pass
