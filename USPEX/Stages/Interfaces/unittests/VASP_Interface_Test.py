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

import numpy as np

from ..VASP_Interface import VASP_Interface
from USPEX.Atomistic.RadialDistributionUtility import RadialDistributionUtility
from USPEX.components import AtomisticRepresentation


HOMEPATH = Path(__file__).parent
SPECIFICPATH = HOMEPATH/'vaspSpecific'
GATHEREDPATH = HOMEPATH/'vaspGatheredData'
WORKPATH = HOMEPATH/'Ca4F8_vasp'


class VASP_CalculatorTest2(unittest.TestCase):
    """
    Checking correct parsing properties
    """
    def test_life(self):
        vasp = VASP_Interface(tag='1', perturbate=False,
                              incar=SPECIFICPATH/'INCAR_1',
                              potcarsPath=SPECIFICPATH,
                              kresol=0.13)
        radialDistributionUtility = RadialDistributionUtility(symbols=['Ca', 'F'])


        for ID in range(10):
            with open(GATHEREDPATH/f'input/system{ID}.vasp', 'rt') as f:
                system = AtomisticRepresentation.readAtomicStructure(f)
                system['ID'] = ID
                system['externalPressure'] = 0.0001
            WORKPATH.mkdir(exist_ok=True)
            vasp.prepareLocalCalculation(system, WORKPATH)
            folder = GATHEREDPATH/'input'/f"CalcFold{system['ID']}"
            dcmp = filecmp.dircmp(folder, WORKPATH)
            match = not dcmp.diff_files
            for common_dir in dcmp.common_dirs:
                match = match and not dcmp.subdirs[common_dir].diff_files
            shutil.rmtree(WORKPATH)
            self.assertTrue(match)
            folder = GATHEREDPATH/'output'
            shutil.copytree(folder/f"CalcFold{system['ID']}", WORKPATH)
            vasp.readOutput(system, WORKPATH)
            shutil.rmtree(WORKPATH)
            with open(folder/f"system{system['ID']}.vasp", 'rt') as f:
                systemRef = AtomisticRepresentation.readAtomicStructure(f)
            self.assertTrue(radialDistributionUtility.equal(system, systemRef))


class VASP_interfaceTest(unittest.TestCase):
    """Read wierd data inside OUTCAR"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.working_dir = HOMEPATH/'wierd_vasp'

    def test1(self):
        wd = self.working_dir
        outcar = wd/'OUTCAR'
        self.interface = VASP_Interface(tag='1',
                                        incar=wd/'Specific'/'INCAR_1',
                                        potcarsPath=wd/'Specific',
                                        kresol=0.05)

        with open(outcar, 'rt') as f:
            content = f.readlines()
        stress = self.interface.readPressureTensor(content)
        assert stress.shape == (3, 3)


class VASP_interface_elastic_Test(unittest.TestCase):

    def test1(self):
        elasticMatrix_ref = [[11184.5135,   609.9831,  1016.7341,  -519.9028,  -109.7639,   -17.9217],
                             [  609.9831,  6321.3677,  1366.9666,   326.5026,   269.2209,    66.3167],
                             [ 1016.7341,  1366.9666,  8253.6909,   -31.0933,   -870.108,    35.496 ],
                             [ -519.9028,   326.5026,   -31.0933,  3193.6427,   308.2938,  -321.5579],
                             [ -109.7639,   269.2209,   -870.108,   308.2938,  1504.6799,   -32.696 ],
                             [  -17.9217,    66.3167,     35.496,  -321.5579,    -32.696,  4247.3728]]
        wd = HOMEPATH/'vaspElastic'
        self.interface = VASP_Interface(tag='5', incar=wd/'Specific'/'INCAR_5', potcarsPath=wd/'Specific',
                                        kresol=0.06, targetProperties=['elasticConstants'])
        system = {}
        self.interface.readOutput(system, calcFolder=wd/'output')
        self.assertTrue(np.allclose(system['elasticMatrix'], elasticMatrix_ref))
