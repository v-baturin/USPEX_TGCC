"""
@file        MLIP_Interface_Test.py
@author:     Michele Galasso
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    m.galasso@yandex.com
@date        10 January 2020
@brief       Class for testing MLIP_Interface.
"""

import shutil
import unittest
import filecmp

from pathlib import Path


from USPEX.Atomistic.RadialDistributionUtility import RadialDistributionUtility
from USPEX.components import AtomisticRepresentation
from ..MLIP_Interface import MLIP_Interface

HOMEPATH = Path(__file__).parent
SPECIFICPATH = HOMEPATH/'mlipSpecific'
GATHEREDPATH = HOMEPATH/'mlipGatheredData'
WORKPATH = HOMEPATH/'NaCl_mlip'


class MLIP_CalculatorTest2(unittest.TestCase):
    """
    Checking correct parsing properties
    """
    def test_life(self):
        mlip = MLIP_Interface(tag='1', input=SPECIFICPATH/'input_1.ini',
                              potential=SPECIFICPATH/'potential.mtp')
        radialDistributionUtility = RadialDistributionUtility(symbols=['Na', 'Cl'])

        for ID in range(10):
            with open(GATHEREDPATH/f'input/system{ID}.vasp', 'rt') as f:
                system = AtomisticRepresentation.readAtomicStructure(f)
            system['externalPressure'] = 100
            system['ID'] = ID
            WORKPATH.mkdir(exist_ok=True)
            mlip.prepareLocalCalculation(system, WORKPATH)
            folder = GATHEREDPATH/'input'/f"CalcFold{system['ID']}"
            dcmp = filecmp.dircmp(folder, WORKPATH)
            match = not dcmp.diff_files
            for common_dir in dcmp.common_dirs:
                match = match and not dcmp.subdirs[common_dir].diff_files
            self.assertTrue(match)
            shutil.rmtree(WORKPATH)
            folder = GATHEREDPATH/'output'
            shutil.copytree(folder/f"CalcFold{system['ID']}", WORKPATH)
            mlip.readOutput(system, WORKPATH)
            shutil.rmtree(WORKPATH)
            with open(folder/f"system{system['ID']}.vasp", 'rt') as f:
                systemRef = AtomisticRepresentation.readAtomicStructure(f)
            self.assertTrue(radialDistributionUtility.equal(system, systemRef))
