"""
@file        MLIP_Interface_Test.py
@author:     Michele Galasso
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    m.galasso@yandex.com
@date        10 January 2020
@brief       Class for testing MLIP_Interface.
"""

import os
import shutil
import unittest
import filecmp

from os.path import join as pj


from ...Atomistic.RadialDistributionUtility import RadialDistributionUtility
from ...components import CrystalRepresentation
from ..MLIP_Interface import MLIP_Interface


HOMEPATH = os.path.dirname(os.path.abspath(__file__))
SPECIFICPATH = pj(HOMEPATH, 'mlipSpecific')
GATHEREDPATH = pj(HOMEPATH, 'mlipGatheredData')
WORKPATH = pj(HOMEPATH, 'NaCl_mlip')


class MLIP_CalculatorTest2(unittest.TestCase):
    """
    Checking correct parsing properties
    """
    def test_life(self):
        mlip = MLIP_Interface(tag='1', input=pj(SPECIFICPATH, 'input_1.ini'),
                              potential=pj(SPECIFICPATH, 'potential.mtp'))
        radialDistributionUtility = RadialDistributionUtility()

        for ID in range(10):
            with open(pj(GATHEREDPATH, f'input/system{ID}.vasp'), 'rt') as f:
                system = CrystalRepresentation.readAtomicStructure(f)
            system['externalPressure'] = 100
            system['ID'] = ID
            os.mkdir(WORKPATH)
            mlip.prepareLocalCalculation(system, WORKPATH)
            folder = pj(GATHEREDPATH, 'input', f"CalcFold{system['ID']}")
            dcmp = filecmp.dircmp(folder, WORKPATH)
            match = not dcmp.diff_files
            for common_dir in dcmp.common_dirs:
                match = match and not dcmp.subdirs[common_dir].diff_files
            self.assertTrue(match)
            shutil.rmtree(WORKPATH)
            folder = pj(GATHEREDPATH, 'output')
            shutil.copytree(pj(folder, f"CalcFold{system['ID']}"), WORKPATH)
            mlip.readOutput(system, WORKPATH)
            shutil.rmtree(WORKPATH)
            with open(pj(folder, f"system{system['ID']}.vasp"), 'rt') as f:
                systemRef = CrystalRepresentation.readAtomicStructure(f)
            self.assertTrue(radialDistributionUtility.equal(system, systemRef))
