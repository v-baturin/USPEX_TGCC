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

from ..MLIP_Interface import MLIP_Interface
from ....DataModel.Flavour import Flavour
from ....Atomistic.Primitives.Element import Element
from ....Atomistic.Primitives.Cell import Cell
from ....Atomistic.Primitives.AtomicStructure import AtomicStructure
from ....IO.AtomicStructureRepresentation import AtomicStructureRepresentation
from ....Atomistic.Atomistic import Atomistic
Atomistic.registerTypes(AtomicStructure, Element, Cell, AtomicStructureRepresentation)



HOMEPATH = Path(__file__).parent
SPECIFICPATH = HOMEPATH/'mlipSpecific'
GATHEREDPATH = HOMEPATH/'mlipGatheredData'
WORKPATH = HOMEPATH/'NaCl_mlip'


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
#             system = AtomicStructureRepresentation.readAtomicStructure(pj(GATHEREDPATH, f'input/system{ID}.vasp'))
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
#             systemRef = AtomicStructureRepresentation.readAtomicStructure(pj(folder, f"system{system['ID']}.vasp"))
#             self.assertTrue(radialDistributionUtility.equal(system, systemRef))


class MLIP_train_Test(unittest.TestCase):

    def setUp(self) -> None:
        self.trainFolder = HOMEPATH/'MLIP_TRAIN'
        self.trainFolder.mkdir()
        shutil.copy(SPECIFICPATH/'24g.mtp', self.trainFolder)
        with open(self.trainFolder/'ts.cfg', 'wt'):
            pass
        self.interface = MLIP_Interface(tag='0', mode='train', potential=self.trainFolder/'24g.mtp',
                                        specorder=['Mo', 'S'], trainingSet=self.trainFolder/'ts.cfg',
                                        args=SPECIFICPATH/'mlip_args_0', sample='trajectory')
        atomistic = Atomistic()
        self.extensions = dict(
            atomistic=atomistic.propertyExtension()
        )


    def test_init(self):
        trajectory = AtomicStructureRepresentation.readMLIPsample(SPECIFICPATH/'configurations.cfg', specorder=['Mo', 'S'])
        intermediate = {'.trajectory': trajectory}
        intermediate = Flavour(extensions=self.extensions, **intermediate)
        calcFolder = HOMEPATH/'MLIP_INIT'
        calcFolder.mkdir(exist_ok=True)
        args = self.interface.prepareLocalCalculation(intermediate, calcFolder=calcFolder)
        self.assertEqual(args, 'train 24g.mtp input.cfg --weight_scaling=2 --weight_scaling_forces=1')
        self.assertTrue(not filecmp.dircmp(HOMEPATH/'MLIP_REF', calcFolder).diff_files)
        shutil.rmtree(calcFolder)


    def test_sample(self):
        intermediate = Flavour(extensions=self.extensions)
        calcFolder=HOMEPATH/'MLIP_REF'
        result = self.interface.readOutput(intermediate, calcFolder=calcFolder)
        self.assertTrue(filecmp.cmp(self.trainFolder/'ts.cfg', calcFolder/'input.cfg'))

    def tearDown(self) -> None:
        shutil.rmtree(self.trainFolder)
