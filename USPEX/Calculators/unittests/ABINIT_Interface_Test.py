"""
@file        ABINIT_Interface_Test.py
@author:     Michele Galasso
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    m.galasso@yandex.com
@date        8 June 2020
@brief       Class for testing ABINIT_Interface class.
"""

import os
import shutil
import unittest
import filecmp

from os.path import join as pj


from ...Atomistic.RadialDistributionUtility import RadialDistributionUtility
from ..ABINIT_Interface import ABINIT_Interface
from ...components import CrystalRepresentation


HOMEPATH = os.path.dirname(os.path.abspath(__file__))
SPECIFICPATH = os.path.join(HOMEPATH, 'abinitSpecific')
GATHEREDPATH = os.path.join(HOMEPATH, 'abinitGatheredData')
WORKPATH = os.path.join(HOMEPATH, 'Eu2H18_abinit')


class ABINIT_Interface_Test(unittest.TestCase):
    """
    Checking correct parsing properties
    """

    def test_life(self):
        abinit = ABINIT_Interface(tag='0', in_file=pj(SPECIFICPATH, 'abinit.in_1'), kresol=0.13,
                                  pp_files=[pj(SPECIFICPATH, 'H.psp8'), pj(SPECIFICPATH, 'Eu.psp8')])
        radialDistributionUtility = RadialDistributionUtility()


        for ID in range(10):
            with open(pj(GATHEREDPATH, f'input/system{ID}.vasp'), 'rt') as f:
                system = CrystalRepresentation.readAtomicStructure(f)
                system['ID'] = ID
                system['externalPressure'] = 130.0
            os.mkdir(WORKPATH)
            abinit.prepareLocalCalculation(system, WORKPATH)
            folder = pj(GATHEREDPATH, 'input', f"CalcFold{system['ID']}")
            dcmp = filecmp.dircmp(folder, WORKPATH)
            match = not dcmp.diff_files
            for common_dir in dcmp.common_dirs:
                match = match and not dcmp.subdirs[common_dir].diff_files
            shutil.rmtree(WORKPATH)
            self.assertTrue(match)
            folder = pj(GATHEREDPATH, 'output')
            shutil.copytree(pj(folder, f"CalcFold{system['ID']}"), WORKPATH)
            abinit.readOutput(system, WORKPATH)
            shutil.rmtree(WORKPATH)
            with open(pj(folder, f"system{system['ID']}.vasp"), 'rt') as f:
                systemRef = CrystalRepresentation.readAtomicStructure(f)
            self.assertTrue(radialDistributionUtility.equal(system, systemRef))
