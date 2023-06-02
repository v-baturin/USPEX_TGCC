'''
@file        PWmat_CalculatorTest.py
@author:     Hao Li
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    samtsevichartem@gmail.com
@date        19 September 2016
@brief       Class for testing VASP_Calculator class.
'''

__author__ = 'Hao Li'

import shutil
import unittest
import numpy as np
from pathlib import Path
from ....components import AtomisticRepresentation, PWmat_Interface


HOMEPATH = Path(__file__).parent
CALC_FOLDER_TEMPLATE = 'CalcFold{}{}'


class PWmat_InterfaceTest(unittest.TestCase):
    '''
    Checking correct parsing properties
    '''

    @classmethod
    def setUpClass(cls):

        cls.knownSystemEnergy = -858.0749767374361

        params = {'tag': 's0', 'kresol': 0.05, 'etot_input': HOMEPATH/'Specific/etot.input_1',
                  'potcars': [HOMEPATH/'Specific/Si.SG15.PBE.UPF']}

        cls.vcEmpty = PWmat_Interface(**params)
        structure = AtomisticRepresentation.readPOSCAR(HOMEPATH/'Si4System.vasp', pbc=(1, 1, 1))
        cls.testSystem = dict(
            ID=0,
            structure=structure,
            disassembler=AtomisticRepresentation.atomicDisassemblerType(
                np.arange(len(structure)).reshape((-1, 1))),
            externalPressure=0.00001
        )

        cls.CALC_FOLDER = HOMEPATH/CALC_FOLDER_TEMPLATE.format(0, 's0')
        cls.REFERENCE_FOLDER = HOMEPATH/'PWmatReference/'

    @classmethod
    def tearDownClass(cls):
        for f in HOMEPATH.iterdir():
            if f.is_dir() and 'CalcFold' in str(f):
                shutil.rmtree(f)

    def test_submit(self):
        self.CALC_FOLDER.mkdir(exist_ok=True)
        self.vcEmpty.prepareLocalCalculation(self.testSystem, self.CALC_FOLDER)

        self.assertTrue(self.CALC_FOLDER.joinpath('etot.input').exists())
        with open(self.CALC_FOLDER/'etot.input') as f:
            content = f.read()
        with open(self.REFERENCE_FOLDER/'etot.input') as f:
            reference = f.read()
        self.assertEqual(content, reference)

        self.assertTrue(self.CALC_FOLDER.joinpath('atom.config').exists())
        with open(self.CALC_FOLDER/'atom.config') as f:
            content = f.read()
        with open(self.REFERENCE_FOLDER/'atom.config') as f:
            reference = f.read()
        self.assertEqual(content, reference)

        #self.vcEmpty.clean(self.testSystem)
        #del self.vcEmpty.submitted[self.testSystem.ID]

    def test_update(self):
        self.POSITIONS_FINAL = np.array(np.matrix(
            '0.000000  0.000000 0.000000;\
            0.000000  0.500000 0.500000;\
            0.500000  0.500000 0.000000;\
            0.500000  0.000000 0.500000;\
            0.750000  0.250000 0.750000;\
            0.250000  0.250000 0.250000;\
            0.250000  0.750000 0.750000;\
            0.750000  0.750000 0.250000'))
        self.LATTICE_FINAL = np.diag([5.48955284] * 3)#np.eye(3) * 4.398995291
        #self.vcEmpty.submitted[self.testSystem.ID] = 1
        self.CALC_FOLDER.mkdir(exist_ok=True)
        shutil.copy(self.REFERENCE_FOLDER/'REPORT', self.CALC_FOLDER)
        shutil.copy(self.REFERENCE_FOLDER/'final.config', self.CALC_FOLDER)
        shutil.copy(self.REFERENCE_FOLDER/'RELAXSTEPS', self.CALC_FOLDER)
        shutil.copy(self.REFERENCE_FOLDER/'MOVEMENT', self.CALC_FOLDER)
        shutil.copy(self.REFERENCE_FOLDER/'IN.RELAXOPT', self.CALC_FOLDER)

        self.assertTrue(self.vcEmpty.isConverged(self.CALC_FOLDER))
        results = self.vcEmpty.readOutput(self.testSystem, self.CALC_FOLDER)
        #self.vcEmpty.clean(self.testSystem)

        structure = results['structure']
        self.assertTrue(np.allclose(self.POSITIONS_FINAL, structure.getCartesianCoordinates(), atol=1.0e-3))
        self.assertTrue(np.allclose(self.LATTICE_FINAL, structure.getCell().getCellVectors(), atol=1.0e-3))
        self.assertAlmostEqual(self.knownSystemEnergy, results['enthalpy'], delta=1.0e-3)
