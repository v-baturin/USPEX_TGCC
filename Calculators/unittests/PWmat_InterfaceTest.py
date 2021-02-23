'''
@file        PWmat_CalculatorTest.py
@author:     Hao Li
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    samtsevichartem@gmail.com
@date        19 September 2016
@brief       Class for testing VASP_Calculator class.
'''

__author__ = 'Hao Li'

import os
import shutil
import unittest
import numpy as np


from ...Atomistic.AtomicStructure import AtomicStructure
from ..PWmat_Interface import  PWmat_Interface
from . import Si4System


HOMEPATH = os.path.dirname(os.path.abspath(__file__))
CALC_FOLDER_TEMPLATE = 'CalcFold{}{}'


class PWmat_InterfaceTest(unittest.TestCase):
    '''
    Checking correct parsing properties
    '''

    @classmethod
    def setUpClass(cls):

        cls.knownSystemEnergy = -858.0749767374361

        tag = Si4System['tag']
        ID = Si4System['ID']
        params = {'tag': tag, 'kresol': 0.05, 'etot_input': '{}/Specific/etot.input_1'.format(HOMEPATH),
                  'potcars': ['{}/Specific/Si.SG15.PBE.UPF'.format(HOMEPATH)]}

        cls.vcEmpty = PWmat_Interface(**params)
        cls.testSystem = {'structure': AtomicStructure.fromDICT(Si4System)}

        cls.CALC_FOLDER = os.path.join(HOMEPATH, CALC_FOLDER_TEMPLATE.format(ID, tag))
        cls.REFERENCE_FOLDER = os.path.join(HOMEPATH , 'Reference/PWmat/')
        print(cls.CALC_FOLDER)
    @classmethod
    def tearDownClass(cls):
        for f in os.listdir(HOMEPATH):
            if os.path.isdir(os.path.join(HOMEPATH, f)) and 'CalcFold' in f:
                shutil.rmtree(os.path.join(HOMEPATH, f))

    def test_submit(self):
        if not os.path.isdir(self.CALC_FOLDER):
            os.mkdir(self.CALC_FOLDER)
        self.vcEmpty.prepareLocalCalculation(self.testSystem, self.CALC_FOLDER)

        self.assertTrue(os.path.exists(os.path.join(self.CALC_FOLDER, 'etot.input')))
        with open(os.path.join(self.CALC_FOLDER , 'etot.input')) as f:
            content = f.read()
        with open(os.path.join(self.REFERENCE_FOLDER , 'etot.input')) as f:
            reference = f.read()
        self.assertEqual(content,reference)

        self.assertTrue(os.path.exists(os.path.join(self.CALC_FOLDER , 'atom.config')))
        with open(os.path.join(self.CALC_FOLDER , 'atom.config')) as f:
            content = f.read()
        with open(os.path.join(self.REFERENCE_FOLDER , 'atom.config')) as f:
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
        if not os.path.isdir(self.CALC_FOLDER):
            os.mkdir(self.CALC_FOLDER)
        shutil.copy(os.path.join(self.REFERENCE_FOLDER , 'REPORT'), self.CALC_FOLDER)
        shutil.copy(os.path.join(self.REFERENCE_FOLDER , 'final.config'), self.CALC_FOLDER)
        shutil.copy(os.path.join(self.REFERENCE_FOLDER , 'RELAXSTEPS'), self.CALC_FOLDER)
        shutil.copy(os.path.join(self.REFERENCE_FOLDER , 'MOVEMENT'), self.CALC_FOLDER)
        shutil.copy(os.path.join(self.REFERENCE_FOLDER, 'IN.RELAXOPT'), self.CALC_FOLDER)

        self.assertTrue(self.vcEmpty.isConverged(self.CALC_FOLDER))
        self.vcEmpty.readOutput(self.testSystem, self.CALC_FOLDER)
        #self.vcEmpty.clean(self.testSystem)
        self.assertTrue(np.allclose(self.POSITIONS_FINAL, self.testSystem['structure'].get_positions(), atol=1.0e-3))
        self.assertTrue(np.allclose(self.LATTICE_FINAL, self.testSystem['structure'].get_cell(), atol=1.0e-3))
        self.assertAlmostEqual(self.knownSystemEnergy, self.testSystem['enthalpy'], delta=1.0e-3)
