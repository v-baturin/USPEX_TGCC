"""
@file        VASP_CalculatorTest.py
@author:     Artem Samtsevich
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    samtsevichartem@gmail.com
@date        19 September 2016
@brief       Class for testing VASP_Calculator class.
"""

__author__ = 'asamtsevich'

import shutil
import unittest
import filecmp

from pathlib import Path

from ..FHIaims_Interface import FHIaims_Interface
from USPEX.Atomistic.RadialDistributionUtility import RadialDistributionUtility
from USPEX.components import AtomisticRepresentation


HOMEPATH = Path(__file__).parent
SPECIFICPATH = HOMEPATH/'aimsSpecific'
GATHEREDPATH = HOMEPATH/'aimsGatheredData'
WORKPATH = HOMEPATH/'F2_aims'


class VASP_CalculatorTest2(unittest.TestCase):
    """
    Checking correct parsing properties
    """
    def test_life(self):
        aims = FHIaims_Interface(tag='1', perturbate=False,
                                 control=SPECIFICPATH/'aims_control_1', kresol=0.14)
        radialDistributionUtility = RadialDistributionUtility(symbols=['P'])


        for ID in range(10):
            with open(GATHEREDPATH/'input'/f'system{ID}.vasp', 'rt') as f:
                system = AtomisticRepresentation.readAtomicStructure(f)
                system['ID'] = ID
                system['externalPressure'] = 0.0001
            WORKPATH.mkdir(exist_ok=True)
            aims.prepareLocalCalculation(system, WORKPATH)
            folder = GATHEREDPATH/'input'/f"CalcFold{system['ID']}"
            dcmp = filecmp.dircmp(folder, WORKPATH)
            match = not dcmp.diff_files
            for common_dir in dcmp.common_dirs:
                match = match and not dcmp.subdirs[common_dir].diff_files
            shutil.rmtree(WORKPATH)
            self.assertTrue(match)
            folder = GATHEREDPATH/'output'
            shutil.copytree(folder/f"CalcFold{system['ID']}", WORKPATH)
            aims.readOutput(system, WORKPATH)
            shutil.rmtree(WORKPATH)
            with open(folder/f"system{system['ID']}.vasp", 'rt') as f:
                systemRef = AtomisticRepresentation.readAtomicStructure(f)
            self.assertTrue(radialDistributionUtility.equal(system, systemRef))

