"""
@file        ABINIT_Interface_Test.py
@author:     Michele Galasso
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    m.galasso@yandex.com
@date        8 June 2020
@brief       Class for testing ABINIT_Interface class.
"""

import shutil
import unittest
import filecmp

from pathlib import Path


from USPEX.Atomistic.RadialDistributionUtility import RadialDistributionUtility
from ..ABINIT_Interface import ABINIT_Interface
from USPEX.components import AtomisticRepresentation


HOMEPATH = Path(__file__).parent
SPECIFICPATH = HOMEPATH/'abinitSpecific'
GATHEREDPATH = HOMEPATH/'abinitGatheredData'
WORKPATH = HOMEPATH/'Eu2H18_abinit'


class ABINIT_Interface_Test(unittest.TestCase):
    """
    Checking correct parsing properties
    """

    def test_life(self):
        abinit = ABINIT_Interface(tag='0', in_file=SPECIFICPATH/'abinit.in_1', kresol=0.13,
                                  pp_files=[SPECIFICPATH/'H.psp8', SPECIFICPATH/'Eu.psp8'])
        radialDistributionUtility = RadialDistributionUtility(symbols=['Eu', 'H'])


        for ID in range(10):
            with open(GATHEREDPATH/f'input/system{ID}.vasp', 'rt') as f:
                system = AtomisticRepresentation.readAtomicStructure(f)
                system['assembled_cell'] = system['cell']
                system['ID'] = ID
                system['externalPressure'] = 130.0
            WORKPATH.mkdir(exist_ok=True)
            abinit.prepareLocalCalculation(system, WORKPATH)
            folder = GATHEREDPATH/'input'/f"CalcFold{system['ID']}"
            dcmp = filecmp.dircmp(folder, WORKPATH)
            match = not dcmp.diff_files
            for common_dir in dcmp.common_dirs:
                match = match and not dcmp.subdirs[common_dir].diff_files
            shutil.rmtree(WORKPATH)
            self.assertTrue(match)
            folder = GATHEREDPATH/'output'
            shutil.copytree(folder/f"CalcFold{system['ID']}", WORKPATH)
            abinit.readOutput(system, WORKPATH)
            shutil.rmtree(WORKPATH)
            with open(folder/f"system{system['ID']}.vasp", 'rt') as f:
                systemRef = AtomisticRepresentation.readAtomicStructure(f)
            self.assertTrue(radialDistributionUtility.equal(system, systemRef))
