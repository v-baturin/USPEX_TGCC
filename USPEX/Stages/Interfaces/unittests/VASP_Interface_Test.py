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

from ....Optimizers.PoolEntry import PoolEntry, EntryFlavour
from ....components import AtomicStructureRepresentation, VASP_Interface, Atomistic


HOMEPATH = Path(__file__).parent
SPECIFICPATH = HOMEPATH/'vaspSpecific'
GATHEREDPATH = HOMEPATH/'vaspGatheredData'
WORKPATH = HOMEPATH/'Ca4F8_vasp'


class VASP_CalculatorTest2(unittest.TestCase):
    """
    Checking correct parsing properties
    """

    def setUp(self) -> None:
        PoolEntry.createEngine(':memory:')

    def test_life(self):
        vasp = VASP_Interface(tag='1', 
                              incar=SPECIFICPATH/'INCAR_1',
                              potcarsPath=SPECIFICPATH,
                              kresol=0.13)
        atomistic = Atomistic()
        extensions = dict(
            atomistic=atomistic.propertyExtension(atomistic)
        )

        for ID in range(10):
            structure = AtomicStructureRepresentation.readPOSCAR(GATHEREDPATH/f'input/system{ID}.vasp', (1, 1, 1))
            disassembler = Atomistic.atomicDisassemblerType(np.arange(len(structure)).reshape((-1, 1)))
            system = PoolEntry(ID, EntryFlavour(extensions=extensions))
            system.setProperty('externalPressure', 0.0001)
            system.setProperty('disassembler', disassembler, extension='atomistic', suffix='intermediate')
            system.setProperty('disassembler', disassembler, extension='atomistic', suffix='1')
            system.setProperty('structure', structure, extension='atomistic', suffix='intermediate')
            WORKPATH.mkdir(exist_ok=True, parents=True)
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
            structureRef = AtomicStructureRepresentation.readPOSCAR(folder/f"system{system['ID']}.vasp", (1, 1, 1))
            structure = system.getProperty('structure', extension='atomistic', suffix='1')
            cell = structure.getCell()
            cellRef = structureRef.getCell()
            self.assertTrue(np.allclose(cell.getCellVectors(),
                                        cellRef.getCellVectors()))
            # self.assertTrue(np.allclose(cell.getWrapedCartesianCoordinates(structure.getCartesianCoordinates()),
            #                             cellRef.getWrapedCartesianCoordinates(structureRef.getCartesianCoordinates())))


class VASP_interfaceTest(unittest.TestCase):
    """Read wierd data inside OUTCAR"""

    @classmethod
    def setUpClass(cls) -> None:
        PoolEntry.createEngine(':memory:')
        cls.working_dir = HOMEPATH/'wierd_vasp'

    def test1(self):
        wd = self.working_dir
        outcar = wd/'OUTCAR'
        self.interface = VASP_Interface(tag='1',
                                        incar=wd/'Specific/INCAR_1',
                                        potcarsPath=wd/'Specific',
                                        kresol=0.05)

        with open(outcar, 'rt') as f:
            content = f.readlines()
        stress = self.interface.readPressureTensor(content)
        assert stress.shape == (3, 3)


class VASP_interface_elastic_Test(unittest.TestCase):

    def setUp(self) -> None:
        PoolEntry.createEngine(':memory:')


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
        with open(wd/'output/OUTCAR', 'r') as f:
            content = f.readlines()
        elasticMatrix = self.interface.readElasticMatrix(content)
        self.assertTrue(np.allclose(elasticMatrix, elasticMatrix_ref))

class VASP_interface_MD_Test(unittest.TestCase):

    def setUp(self) -> None:
        PoolEntry.createEngine(':memory:')


    def test1(self):
        wd = HOMEPATH/'AIMD_AlB2'
        self.interface = VASP_Interface(tag='1', incar=wd/'INCAR', potcarsPath=wd,
                                        kresol=0.06, targetProperties=['trajectory'])
        atomistic = Atomistic()
        extensions = dict(
            atomistic=atomistic.propertyExtension(atomistic)
        )

        system = PoolEntry(0, EntryFlavour(extensions=extensions))
        self.interface.readOutput(system, wd)
        self.assertGreater(len(system['.trajectory.1']), 1)
        for data in system['.trajectory.1']:
            self.assertTrue(len(data['structure']) == 3)
            self.assertTrue('energy' in data['results'])
            self.assertTrue('forces' in data['results'])
