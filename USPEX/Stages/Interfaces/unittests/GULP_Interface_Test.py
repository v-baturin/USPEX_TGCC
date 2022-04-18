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

from USPEX.Atomistic.RadialDistributionUtility import RadialDistributionUtility
from USPEX.components import AtomisticRepresentation, GULP_Interface


HOMEPATH = os.path.dirname(os.path.abspath(__file__))
SPECIFICPATH = pj(HOMEPATH, 'gulpSpecific')
GATHEREDPATH = pj(HOMEPATH, 'gulpGatheredData')
WORKPATH = pj(HOMEPATH, 'Mg4Al8O16_gulp')


class GULP_CalculatorTest(unittest.TestCase):


    def test_life(self):

        gulp = GULP_Interface(tag='0', perturbate=False,
                              goptions=pj(SPECIFICPATH, 'goptions'), ginput=pj(SPECIFICPATH, 'ginput_1'))
        radialDistributionUtility = RadialDistributionUtility()

        for ID in range(10):
            with open(pj(GATHEREDPATH, f'input/system{ID}.vasp'), 'rt') as f:
                system = AtomisticRepresentation.readAtomicStructure(f)
            system['externalPressure'] = 100
            system['ID'] = ID
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
            with open(pj(folder, f"system{system['ID']}.vasp"), 'rt') as f:
                systemRef = AtomisticRepresentation.readAtomicStructure(f)
            self.assertTrue(radialDistributionUtility.equal(system, systemRef))


class GULP_InterfaceTest(unittest.TestCase):
    def test_read_output(self):
        ID = 0
        # HERE what is written in ginput and goption no make sense.
        # Only output will be parsed and properties checked
        interface = GULP_Interface(tag='1', ginput=pj(HOMEPATH, 'Specific', 'ginput_1'),
                                   goptions=pj(HOMEPATH, 'Specific', 'goptions_1'),
                                   targetProperties=['structure', 'enthalpy', 'stressTensor', 'strains'])
        # with open(pj(GATHEREDPATH, f'input/system{ID}'), 'rt') as f:
        #     system = {'ID': ID, 'structure': Crystal.fromJSON(f.read())}
        with open(pj(GATHEREDPATH, f'input/system{ID}.vasp'), 'rt') as f:
            system = AtomisticRepresentation.readAtomicStructure(f)
        system['assembledCell'] = system['cell']
        system['ID'] = 0
        system['disassembler'] = AtomisticRepresentation.atomicDisassemblerType.createFlatDisassembler(len(system['molecules']), cell=system['cell'])

        interface.readOutput(system=system, calcFolder=pj(HOMEPATH, 'gulp_test'))
        self.assertTrue(np.isclose(system['enthalpy'], -645.80329121))
        stress_ref = np.array([[-99.960848, 0.394719, -0.211383], [0.394719,  -100.008335,  0.019149], [-0.211383,  0.019149,  -100.271584]])
        self.assertTrue(np.allclose(system['stressTensor'], stress_ref))
        strains_ref = np.array([0.012703, -0.031870, -0.039885, -0.000385, -0.008334, 0.182955])
        self.assertTrue(np.allclose(system['strains'], strains_ref))

    def test_read_energy(self):
         with open(pj(HOMEPATH, 'gulp_test', 'output_bad_1st_SCF'), 'rt') as f:
             contents = f.readlines()
         ev = GULP_Interface.readEnergy(None, contents)
         self.assertAlmostEqual(ev, -1604.1694, delta=0.001)

