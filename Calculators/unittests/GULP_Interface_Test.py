"""
@file        GULP_CalculatorTest.py
@author:     Artem Samtsevich
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    samtsevichartem@gmail.com
@date        19 September 2016
@brief       Class for testing GULP_Calculator class.
"""

__author__ = 'asamtsevich'

import numpy as np
import os
import shutil
import unittest
import filecmp

from os.path import join as pj

from ...Atomistic.Crystal import Crystal

from ..GULP_Interface import GULP_Interface


HOMEPATH = os.path.dirname(os.path.abspath(__file__))
SPECIFICPATH = pj(HOMEPATH, 'gulpSpecific')
GATHEREDPATH = pj(HOMEPATH, 'gulpGatheredData')
WORKPATH = pj(HOMEPATH, 'Mg4Al8O16_gulp')


class GULP_CalculatorTest(unittest.TestCase):


    def test_life(self):
        config = {'externalPressure' : 100}

        gulp = GULP_Interface(tag='0', perturbate=False,
                              goptions=pj(SPECIFICPATH, 'goptions'), ginput=pj(SPECIFICPATH, 'ginput_1'))

        for ID in range(10):
            with open(pj(GATHEREDPATH, f'input/system{ID}'), 'rt') as f:
                system = {'ID': ID, 'structure': Crystal.fromJSON(f.read())}
            system['structure'].config = config
            os.mkdir(WORKPATH)
            gulp.prepareLocalCalculation(system, WORKPATH)
            folder = pj(GATHEREDPATH, 'input', f"CalcFold{system['ID']}")
            dcmp = filecmp.dircmp(folder, WORKPATH)
            match = not dcmp.diff_files
            for common_dir in dcmp.common_dirs:
                match = match and not dcmp.subdirs[common_dir].diff_files
            shutil.rmtree(WORKPATH)
            self.assertTrue(match)
            folder = pj(GATHEREDPATH, 'output')
            shutil.copytree(pj(folder, f"CalcFold{system['ID']}"), WORKPATH)
            gulp.readOutput(system, WORKPATH)
            shutil.rmtree(WORKPATH)
            system['structure']._fingerprint = None
            with open(pj(folder, f"system{system['ID']}"), 'rt') as f:
                systemRef = system['structure'].fromJSON(f.read())
            self.assertEqual(system['structure'], systemRef)


class GULP_InterfaceTest(unittest.TestCase):
    def test_read_output(self):
        ID = 0
        # HERE what is written in ginput and goption no make sense.
        # Only output will be parsed and properties checked
        interface = GULP_Interface(tag='1', ginput=pj(HOMEPATH, 'Specific', 'ginput_1'),
                                            goptions=pj(HOMEPATH, 'Specific', 'goptions_1'))
        with open(pj(GATHEREDPATH, f'input/system{ID}'), 'rt') as f:
            system = {'ID': ID, 'structure': Crystal.fromJSON(f.read())}

        interface.readOutput(system=system, calcFolder=pj(HOMEPATH, 'gulp_test'))
        self.assertTrue(np.isclose(system['enthalpy'], -645.80329121))
        stress_ref = np.array([[-99.960848, 0.394719, -0.211383], [0.394719,  -100.008335,  0.019149], [-0.211383,  0.019149,  -100.271584]])
        self.assertTrue(np.allclose(system['stressTensor'], stress_ref))
        strains_ref = np.array([0.012703, -0.031870, -0.039885, -0.000385, -0.008334, 0.182955])
        self.assertTrue(np.allclose(system['strains'], strains_ref))

