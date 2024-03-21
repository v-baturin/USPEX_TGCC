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
import numpy as np

from pathlib import Path

from ..ABINIT_Interface import ABINIT_Interface
from ....Optimizers.PoolEntry import EntryFlavour
from ....Atomistic.Primitives.Element import Element
from ....Atomistic.Primitives.Cell import Cell
from ....Atomistic.Primitives.AtomicStructure import AtomicStructure
from ....IO.AtomicStructureRepresentation import AtomicStructureRepresentation
from ....Atomistic.Atomistic import Atomistic
Atomistic.registerTypes(AtomicStructure, Element, Cell, AtomicStructureRepresentation)


HOMEPATH = Path(__file__).parent
SPECIFICPATH = HOMEPATH/'abinitSpecific'
GATHEREDPATH = HOMEPATH/'abinitGatheredData'
WORKPATH = HOMEPATH/'Eu2H18_abinit'

try:
    from abipy import abilab
except Exception:
    pass
else:
    class ABINIT_Interface_Test(unittest.TestCase):
        """
        Checking correct parsing properties
        """

        def test_life(self):
            abinit = ABINIT_Interface(tag='0',
                                      in_file=SPECIFICPATH/'abinit.in_1',
                                      kresol=0.13,
                                      pp_files=[SPECIFICPATH/'H.psp8', SPECIFICPATH/'Eu.psp8'],
                                      targetProperties=['structure', 'enthalpy'])
            atomistic = Atomistic()
            extensions = dict(
                atomistic=atomistic.propertyExtension()
            )

            for ID in range(10):
                structure = AtomicStructureRepresentation.readPOSCAR(GATHEREDPATH/f'input/system{ID}.vasp', (1, 1, 1))
                disassembler = Atomistic.atomicDisassemblerType(np.arange(len(structure)).reshape((-1, 1)))
                intermediate = disassembler.disassemble(structure)
                intermediate['.externalPressure'] = 130
                intermediate['atomistic.disassembler'] = disassembler
                intermediate = EntryFlavour(extensions=extensions, **intermediate)
                WORKPATH.mkdir(parents=True, exist_ok=True)
                abinit.prepareLocalCalculation(intermediate, WORKPATH)
                folder = GATHEREDPATH/'input'/f"CalcFold{ID}"
                dcmp = filecmp.dircmp(folder, WORKPATH)
                match = not dcmp.diff_files
                for common_dir in dcmp.common_dirs:
                    match = match and not dcmp.subdirs[common_dir].diff_files
                shutil.rmtree(WORKPATH)
                self.assertTrue(match)
                folder = GATHEREDPATH/'output'
                shutil.copytree(folder/f"CalcFold{ID}", WORKPATH)
                result = abinit.readOutput(intermediate, WORKPATH)
                shutil.rmtree(WORKPATH)
                structureRef = AtomicStructureRepresentation.readPOSCAR(folder/f"system{ID}.vasp", (1, 1, 1))
                cell = result['atomistic.structure'].getCell()
                cellRef = structureRef.getCell()
                self.assertTrue(np.allclose(cell.getCellVectors(),
                                            cellRef.getCellVectors()))
                # self.assertTrue(np.allclose(cell.getWrapedCartesianCoordinates(results['structure'].getCartesianCoordinates()),
                #                             cellRef.getWrapedCartesianCoordinates(structureRef.getCartesianCoordinates())))
