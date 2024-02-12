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

from ..PHONOPY_Interface import PHONOPY_Interface
from ....Optimizers.PoolEntry import EntryFlavour
from ....Atomistic.Primitives.Element import Element
from ....Atomistic.Primitives.Cell import Cell
from ....Atomistic.Primitives.AtomicStructure import AtomicStructure
from ....IO.AtomicStructureRepresentation import AtomicStructureRepresentation
from ....Atomistic.Atomistic import Atomistic
Atomistic.registerTypes(AtomicStructure, Element, Cell, AtomicStructureRepresentation)
PHONOPY_Interface.registerTypes(AtomicStructureRepresentation)


HOMEPATH = Path(__file__).parent
SPECIFICPATH = HOMEPATH/'vaspSpecific'
GATHEREDPATH = HOMEPATH/'vaspGatheredData'
PHONOPYSPECIFIC = HOMEPATH / 'phonopySpecific'
WORKPATH = HOMEPATH/'PHONOPY_dir'


class PHONOPY_interfaceTest0(unittest.TestCase):
    """
    Checking correct parsing properties
    """

    def test_life(self):
        phonopy = PHONOPY_Interface(tag='1',
                              incar=SPECIFICPATH/'INCAR_1',
                              potcarsPath=SPECIFICPATH,
                              phRunscriptTemplatePath=PHONOPYSPECIFIC/'script_phonopy.sh',
                              kresol=0.13,
                              targetProperties=['zpe'])
        atomistic = Atomistic()
        extensions = dict(
            atomistic=atomistic.propertyExtension(atomistic)
        )

        ID = 0

        structure = AtomicStructureRepresentation.readPOSCAR(GATHEREDPATH/f'input/system{ID}.vasp', (1, 1, 1))
        disassembler = Atomistic.atomicDisassemblerType(np.arange(len(structure)).reshape((-1, 1)))
        intermediate = disassembler.disassemble(structure)
        intermediate['.externalPressure'] = 0.0001
        intermediate['.ID'] = ID
        intermediate['atomistic.disassembler'] = disassembler
        intermediate = EntryFlavour(extensions=extensions, **intermediate)
        WORKPATH.mkdir(exist_ok=True, parents=True)
        phonopy.prepareLocalCalculation(intermediate, WORKPATH)
        # folder = GATHEREDPATH/'input'/f"CalcFold{ID}"
        # dcmp = filecmp.dircmp(folder, WORKPATH)
        # match = not dcmp.diff_files
        # for common_dir in dcmp.common_dirs:
        #     match = match and not dcmp.subdirs[common_dir].diff_files
        # shutil.rmtree(WORKPATH)
        # self.assertTrue(match)
        # folder = GATHEREDPATH/'output'
        # shutil.copytree(folder/f"CalcFold{ID}", WORKPATH)
        # result = vasp.readOutput(intermediate, WORKPATH)
        # shutil.rmtree(WORKPATH)
        # structureRef = AtomicStructureRepresentation.readPOSCAR(folder/f"system{ID}.vasp", (1, 1, 1))
        # structure = result.getProperty('structure', extension='atomistic')
        # cell = structure.getCell()
        # cellRef = structureRef.getCell()
        # self.assertTrue(np.allclose(cell.getCellVectors(),
        #                             cellRef.getCellVectors()))
        # self.assertTrue(np.allclose(cell.getWrapedCartesianCoordinates(structure.getCartesianCoordinates()),
        #                             cellRef.getWrapedCartesianCoordinates(structureRef.getCartesianCoordinates())))


# class PHONOPY_interfaceTest(unittest.TestCase):
#     """Read wierd data inside OUTCAR"""
#
#     @classmethod
#     def setUpClass(cls) -> None:
#         cls.working_dir = HOMEPATH/'PHONOPY_dir'
#
#     def test1(self):
#         wd = self.working_dir
#         outcar = wd/'OUTCAR'
#         self.interface = VASP_Interface(tag='1',
#                                         incar=wd/'Specific/INCAR_1',
#                                         potcarsPath=wd/'Specific',
#                                         kresol=0.05,
#                                         targetProperties=['structure', 'enthalpy'])
#
#         with open(outcar, 'rt') as f:
#             content = f.readlines()
#         stress = self.interface.readPressureTensor(content)
#         assert stress.shape == (3, 3)
#
#
# class VASP_interface_elastic_Test(unittest.TestCase):
#
#     def test1(self):
#         elasticMatrix_ref = [[11184.5135,   609.9831,  1016.7341,  -519.9028,  -109.7639,   -17.9217],
#                              [  609.9831,  6321.3677,  1366.9666,   326.5026,   269.2209,    66.3167],
#                              [ 1016.7341,  1366.9666,  8253.6909,   -31.0933,   -870.108,    35.496 ],
#                              [ -519.9028,   326.5026,   -31.0933,  3193.6427,   308.2938,  -321.5579],
#                              [ -109.7639,   269.2209,   -870.108,   308.2938,  1504.6799,   -32.696 ],
#                              [  -17.9217,    66.3167,     35.496,  -321.5579,    -32.696,  4247.3728]]
#         wd = HOMEPATH/'vaspElastic'
#         self.interface = VASP_Interface(tag='5', incar=wd/'Specific'/'INCAR_5', potcarsPath=wd/'Specific',
#                                         kresol=0.06, targetProperties=['elasticConstants'])
#         with open(wd/'output/OUTCAR', 'r') as f:
#             content = f.readlines()
#         elasticMatrix = self.interface.readElasticMatrix(content)
#         self.assertTrue(np.allclose(elasticMatrix, elasticMatrix_ref))
#
# class VASP_interface_MD_Test(unittest.TestCase):
#
#     def test1(self):
#         wd = HOMEPATH/'AIMD_AlB2'
#         self.interface = VASP_Interface(tag='1', incar=wd/'INCAR', potcarsPath=wd,
#                                         kresol=0.06, targetProperties=['trajectory'])
#         atomistic = Atomistic()
#         extensions = dict(
#             atomistic=atomistic.propertyExtension(atomistic)
#         )
#
#         result = self.interface.readOutput(EntryFlavour(extensions=extensions), wd)
#         self.assertGreater(len(result['.trajectory']), 1)
#         for data in result['.trajectory']:
#             self.assertTrue(len(data['structure']) == 3)
#             self.assertTrue('energy' in data['results'])
#             self.assertTrue('forces' in data['results'])
