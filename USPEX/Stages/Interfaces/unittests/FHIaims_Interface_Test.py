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
import numpy as np

from pathlib import Path

from ..FHIaims_Interface import FHIaims_Interface
from ....Optimizers.PoolEntry import EntryFlavour
from ....Atomistic.Primitives.Element import Element
from ....Atomistic.Primitives.Cell import Cell
from ....Atomistic.Primitives.AtomicStructure import AtomicStructure
from ....IO.AtomicStructureRepresentation import AtomicStructureRepresentation
from ....Atomistic.Atomistic import Atomistic
Atomistic.registerTypes(AtomicStructure, Element, Cell, AtomicStructureRepresentation)



HOMEPATH = Path(__file__).parent
SPECIFICPATH = HOMEPATH/'aimsSpecific'
GATHEREDPATH = HOMEPATH/'aimsGatheredData'
WORKPATH = HOMEPATH/'F2_aims'


class VASP_CalculatorTest2(unittest.TestCase):
    """
    Checking correct parsing properties
    """

    def test_life(self):
        aims = FHIaims_Interface(tag='1',
                                 control=SPECIFICPATH/'aims_control_1',
                                 kresol=0.14, targetProperties=['structure', 'enthalpy'])
        atomistic = Atomistic()
        extensions = dict(
            atomistic=atomistic.propertyExtension(atomistic)
        )

        for ID in range(10):
            structure = AtomicStructureRepresentation.readPOSCAR(GATHEREDPATH/f'input/system{ID}.vasp', (1, 1, 1))
            disassembler = Atomistic.atomicDisassemblerType(np.arange(len(structure)).reshape((-1, 1)))
            intermediate = disassembler.disassemble(structure)
            intermediate['.externalPressure'] = 0.0001
            intermediate['atomistic.disassembler'] = disassembler
            intermediate = EntryFlavour(extensions=extensions, **intermediate)
            WORKPATH.mkdir(parents=True, exist_ok=True)
            aims.prepareLocalCalculation(intermediate, WORKPATH)
            folder = GATHEREDPATH/'input'/f"CalcFold{ID}"
            dcmp = filecmp.dircmp(folder, WORKPATH)
            match = not dcmp.diff_files
            for common_dir in dcmp.common_dirs:
                match = match and not dcmp.subdirs[common_dir].diff_files
            shutil.rmtree(WORKPATH)
            self.assertTrue(match)
            folder = GATHEREDPATH/'output'
            shutil.copytree(folder/f"CalcFold{ID}", WORKPATH)
            result = aims.readOutput(intermediate, WORKPATH)
            shutil.rmtree(WORKPATH)
            structureRef = AtomicStructureRepresentation.readPOSCAR(folder/f"system{ID}.vasp", (1, 1, 1))
            structure = result.getProperty('structure', extension='atomistic')
            cell = structure.getCell()
            cellRef = structureRef.getCell()
            self.assertTrue(np.allclose(cell.getCellVectors(),
                                        cellRef.getCellVectors()))
            self.assertTrue(np.allclose(cell.getWrapedCartesianCoordinates(structure.getCartesianCoordinates()),
                                        cellRef.getWrapedCartesianCoordinates(structureRef.getCartesianCoordinates())))

