import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import unittest
import os
import json

from USPEX.Common.Atomistic.mol.read_molecule import read_molecule
from USPEX.Common.Atomistic.Crystal import Crystal
from USPEX.Common.Atomistic.CrystalPool import CrystalPool
from ..Rotation import Rotation, VOFailed


HOMEPATH = os.path.dirname(os.path.abspath(__file__))


class Rotation_Test(unittest.TestCase):

    def test_molecular_fixed(self):
        mol_glycine = read_molecule(f'{HOMEPATH}/MOL_glycine')
        config = {}
        pool = CrystalPool(symbols = [mol_glycine], blocks = [[4]], range = [[1, 1]])
        rotation = Rotation(Crystal, config, pool, initFrac=1.0)

        with open(f'{HOMEPATH}/molecular_structures_fixed', 'rt') as f:
            population = json.loads(f.read())
        population_iterator = iter(population)
        count = 0
        while count < 1:
            try:
                parent = Crystal.fromDICT(next(population_iterator))
                offsprings = rotation(parent)
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
        pool = CrystalPool(symbols = [mol_1, mol_2, mol_3, 'O', mol_5],
                           blocks = [[0,2,0,1,0], [1,0,0,0,0],[0,0,1,0,0],[0,0,0,0,1]],
                           range = [[1,2],[0,4],[0,4],[0,4]], minAt = 5, maxAt = 40)
        rotation = Rotation(Crystal, config, pool, initFrac=1.0)

        with open(f'{HOMEPATH}/molecular_structures_variable', 'rt') as f:
            population = json.loads(f.read())
        population_iterator = iter(population)
        count = 0
        while count < 1:
            try:
                parent = Crystal.fromDICT(next(population_iterator))
                parent.config = config
                offsprings = rotation(parent)
                count += len(offsprings)
            except VOFailed:
                pass
