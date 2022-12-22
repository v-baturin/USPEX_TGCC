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


from USPEX.Atomistic.RadialDistributionUtility import RadialDistributionUtility
from USPEX.components import AtomisticRepresentation
from ..MLIP_Interface import MLIP_Interface


HOMEPATH = os.path.dirname(os.path.abspath(__file__))
SPECIFICPATH = pj(HOMEPATH, 'mlipSpecific')
GATHEREDPATH = pj(HOMEPATH, 'mlipGatheredData')
WORKPATH = pj(HOMEPATH, 'NaCl_mlip')


# class MLIP_CalculatorTest2(unittest.TestCase):
#     """
#     Checking correct parsing properties
#     """
#     def test_life(self):
#         mlip = MLIP_Interface(tag='1', input=pj(SPECIFICPATH, 'input_1.ini'),
#                               potential=pj(SPECIFICPATH, 'potential.mtp'))
#         radialDistributionUtility = RadialDistributionUtility(symbols=['Na', 'Cl'])
#
#         for ID in range(10):
#             system = AtomisticRepresentation.readAtomicStructure(pj(GATHEREDPATH, f'input/system{ID}.vasp'))
#             system['externalPressure'] = 100
#             system['ID'] = ID
#             system['tmp_1'] = {}
#             os.mkdir(WORKPATH)
#             mlip.prepareLocalCalculation(system, WORKPATH)
#             folder = pj(GATHEREDPATH, 'input', f"CalcFold{system['ID']}")
#             dcmp = filecmp.dircmp(folder, WORKPATH)
#             match = not dcmp.diff_files
#             for common_dir in dcmp.common_dirs:
#                 match = match and not dcmp.subdirs[common_dir].diff_files
#             self.assertTrue(match)
#             shutil.rmtree(WORKPATH)
#             folder = pj(GATHEREDPATH, 'output')
#             shutil.copytree(pj(folder, f"CalcFold{system['ID']}"), WORKPATH)
#             mlip.readOutput(system, WORKPATH)
#             shutil.rmtree(WORKPATH)
#             systemRef = AtomisticRepresentation.readAtomicStructure(pj(folder, f"system{system['ID']}.vasp"))
#             self.assertTrue(radialDistributionUtility.equal(system, systemRef))


class MLIP_train_Test(unittest.TestCase):

    def setUp(self) -> None:
        self.trainFolder = pj(HOMEPATH, 'MLIP_TRAIN')
        os.mkdir(self.trainFolder)
        shutil.copy2(pj(SPECIFICPATH, '24g.mtp'), self.trainFolder)
        self.interface = MLIP_Interface(tag='0', mode='train', potential=pj(self.trainFolder, '24g.mtp'),
                                        specorder=['Mo', 'S'], trainingSet=pj(self.trainFolder, 'ts.cfg'),
                                        args=pj(SPECIFICPATH, 'mlip_args_0'))

    def test_init(self):
        system = dict(
            ID=0,
            tmp_0={}
        )
        system['mlip.sample'] = AtomisticRepresentation.readMLIPsample(pj(SPECIFICPATH, 'configurations.cfg'),
                                                                       specorder=['Mo', 'S'])
        calcFolder = pj(HOMEPATH, 'MLIP_INIT')
        os.mkdir(calcFolder)
        args = self.interface.prepareLocalCalculation(system=system, calcFolder=calcFolder)
        self.assertEqual(args, 'train 24g.mtp input.cfg --weight_scaling=2 --weight_scaling_forces=1')
        self.assertTrue(not filecmp.dircmp(pj(HOMEPATH, 'MLIP_REF'), calcFolder).diff_files)
        shutil.rmtree(calcFolder)


    def test_sample(self):
        system = dict(
            tmp_0={}
        )

        calcFolder=pj(HOMEPATH, 'MLIP_REF')
        self.interface.readOutput(system=system, calcFolder=calcFolder)
        self.assertTrue(filecmp.cmp(pj(self.trainFolder, 'ts.cfg'), pj(calcFolder, 'input.cfg')))

    def tearDown(self) -> None:
        shutil.rmtree(self.trainFolder)
