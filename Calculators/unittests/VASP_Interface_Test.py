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


from ...Atomistic.Crystal import Crystal
from ..VASP_Interface import VASP_Interface


HOMEPATH = os.path.dirname(os.path.abspath(__file__))
SPECIFICPATH = pj(HOMEPATH, 'vaspSpecific')
GATHEREDPATH = pj(HOMEPATH, 'vaspGatheredData')
WORKPATH = pj(HOMEPATH, 'Ca4F8_vasp')


class VASP_CalculatorTest2(unittest.TestCase):
    """
    Checking correct parsing properties
    """
    def test_life(self):
        vasp = VASP_Interface(tag='1', incar=pj(SPECIFICPATH, 'INCAR_1'), potcarsPath=SPECIFICPATH, kresol=0.13)

        for ID in range(10):
            with open(pj(GATHEREDPATH, f'input/system{ID}'), 'rt') as f:
                system = {'ID': ID, 'structure': Crystal.fromJSON(f.read())}
            os.mkdir(WORKPATH)
            vasp.prepareLocalCalculation(system, WORKPATH)
            folder = pj(GATHEREDPATH, 'input', f"CalcFold{system['ID']}")
            dcmp = filecmp.dircmp(folder, WORKPATH)
            match = not dcmp.diff_files
            for common_dir in dcmp.common_dirs:
                match = match and not dcmp.subdirs[common_dir].diff_files
            shutil.rmtree(WORKPATH)
            self.assertTrue(match)
            folder = pj(GATHEREDPATH, 'output')
            shutil.copytree(pj(folder, f"CalcFold{system['ID']}"), WORKPATH)
            vasp.readOutput(system, WORKPATH)
            shutil.rmtree(WORKPATH)
            system['structure']._fingerprint = None
            with open(pj(folder, f"system{system['ID']}"), 'rt') as f:
                systemRef = system['structure'].fromJSON(f.read())
            self.assertEqual(system['structure'], systemRef)


class VASP_interfaceTest(unittest.TestCase):
    """Read wierd data inside OUTCAR"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.working_dir = pj(HOMEPATH, 'wierd_vasp')

    def test1(self):
        wd = self.working_dir
        outcar = pj(wd, 'OUTCAR')
        self.interface = VASP_Interface(tag='1', incar=pj(wd, 'Specific', 'INCAR_1'), potcarsPath=pj(wd, 'Specific'),
                                        kresol=0.05)

        stress = self.interface.readPressureTensor(outcar)
        assert stress.shape == (3, 3)
