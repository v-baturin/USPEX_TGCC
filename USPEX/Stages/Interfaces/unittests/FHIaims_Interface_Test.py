"""
@file        VASP_CalculatorTest.py
@author:     Artem Samtsevich
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    samtsevichartem@gmail.com
@date        19 September 2016
@brief       Class for testing VASP_Calculator class.
"""

__author__ = 'asamtsevich'

import os
import shutil
import unittest
import filecmp

from os.path import join as pj


from USPEX.Atomistic.RadialDistributionUtility import RadialDistributionUtility
from ..FHIaims_Interface import FHIaims_Interface
from USPEX.components import AtomisticRepresentation


HOMEPATH = os.path.dirname(os.path.abspath(__file__))
SPECIFICPATH = pj(HOMEPATH, 'aimsSpecific')
GATHEREDPATH = pj(HOMEPATH, 'aimsGatheredData')
WORKPATH = pj(HOMEPATH, 'F2_aims')


class VASP_CalculatorTest2(unittest.TestCase):
    """
    Checking correct parsing properties
    """
    def test_life(self):
        aims = FHIaims_Interface(tag='1', perturbate=False,
                              control=pj(SPECIFICPATH, 'aims_control_1'), kresol=0.14)
        radialDistributionUtility = RadialDistributionUtility()


        for ID in range(10):
            with open(pj(GATHEREDPATH, f'input/system{ID}.vasp'), 'rt') as f:
                system = AtomisticRepresentation.readAtomicStructure(f)
                system['ID'] = ID
                system['externalPressure'] = 0.0001
            os.mkdir(WORKPATH)
            aims.prepareLocalCalculation(system, WORKPATH)
            folder = pj(GATHEREDPATH, 'input', f"CalcFold{system['ID']}")
            dcmp = filecmp.dircmp(folder, WORKPATH)
            match = not dcmp.diff_files
            for common_dir in dcmp.common_dirs:
                match = match and not dcmp.subdirs[common_dir].diff_files
            shutil.rmtree(WORKPATH)
            self.assertTrue(match)
            folder = pj(GATHEREDPATH, 'output')
            shutil.copytree(pj(folder, f"CalcFold{system['ID']}"), WORKPATH)
            aims.readOutput(system, WORKPATH)
            shutil.rmtree(WORKPATH)
            with open(pj(folder, f"system{system['ID']}.vasp"), 'rt') as f:
                systemRef = AtomisticRepresentation.readAtomicStructure(f)
            self.assertTrue(radialDistributionUtility.equal(system, systemRef))

