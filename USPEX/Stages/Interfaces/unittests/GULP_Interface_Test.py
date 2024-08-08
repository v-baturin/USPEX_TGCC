"""
@file        GULP_CalculatorTest.py
@author:     Artem Samtsevich
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    samtsevichartem@gmail.com
@date        19 September 2016
@brief       Class for testing GULP_Calculator class.
"""

__author__ = 'asamtsevich'

import numpy as np
import shutil
import unittest
import filecmp

from pathlib import Path

from ..GULP_Interface import GULP_Interface
from ....DataModel.Flavour import Flavour
from ....Atomistic.Primitives.Element import Element
from ....Atomistic.Primitives.Cell import Cell
from ....Atomistic.Primitives.AtomicStructure import AtomicStructure
from ....IO.AtomicStructureRepresentation import AtomicStructureRepresentation
from ....Atomistic.Atomistic import Atomistic
Atomistic.registerTypes(AtomicStructure, Element, Cell, AtomicStructureRepresentation)


HOMEPATH = Path(__file__).parent
SPECIFICPATH = HOMEPATH/'gulpSpecific'
GATHEREDPATH = HOMEPATH/'gulpGatheredData'
WORKPATH = HOMEPATH/'Mg4Al8O16_gulp'


class GULP_CalculatorTest(unittest.TestCase):

    def test_life(self):

        gulp = GULP_Interface(tag='0', goptions=SPECIFICPATH/'goptions', ginput=SPECIFICPATH/'ginput_1',
                              targetProperties=['structure', 'enthalpy'])
        atomistic = Atomistic()
        extensions = dict(
            atomistic=(atomistic, atomistic.propertyExtension.propertyTable),
        )

        for ID in range(10):
            structure = AtomicStructureRepresentation.readPOSCAR(GATHEREDPATH/f'input/system{ID}.vasp', (1, 1, 1))
            disassembler = Atomistic.atomicDisassemblerType(np.arange(len(structure)).reshape((-1, 1)))
            intermediate = disassembler.disassemble(structure)
            intermediate['.externalPressure'] = 100
            intermediate['atomistic.disassembler'] = disassembler
            intermediate = Flavour(extensions=extensions, **intermediate)
            WORKPATH.mkdir(parents=True, exist_ok=True)
            gulp.prepareLocalCalculation(intermediate, WORKPATH)
            folder = GATHEREDPATH/'input'/f"CalcFold{ID}"
            dcmp = filecmp.dircmp(folder, WORKPATH)
            match = not dcmp.diff_files
            for common_dir in dcmp.common_dirs:
                match = match and not dcmp.subdirs[common_dir].diff_files
            shutil.rmtree(WORKPATH)
            self.assertTrue(match)
            folder = GATHEREDPATH/'output'
            shutil.copytree(folder/f"CalcFold{ID}", WORKPATH)
            result = gulp.readOutput(intermediate, WORKPATH)
            shutil.rmtree(WORKPATH)
            structureRef = AtomicStructureRepresentation.readPOSCAR(folder/f"system{ID}.vasp", (1, 1, 1))
            structure = result.getProperty('structure', extension='atomistic')
            cell = structure.getCell()
            cellRef = structureRef.getCell()
            self.assertTrue(np.allclose(cell.getCellVectors(),
                                        cellRef.getCellVectors()))
            # self.assertTrue(np.allclose(cell.getWrapedCartesianCoordinates(results['structure'].getCartesianCoordinates()),
            #                             cellRef.getWrapedCartesianCoordinates(structureRef.getCartesianCoordinates())))


class GULP_InterfaceTest(unittest.TestCase):

    def test_read_output(self):
        ID = 0
        # HERE what is written in ginput and goption no make sense.
        # Only output will be parsed and properties checked
        interface = GULP_Interface(tag='1', ginput=HOMEPATH/'Specific'/'ginput_1',
                                   goptions= HOMEPATH/'Specific'/'goptions_1',
                                   targetProperties=['structure', 'enthalpy', 'stressTensor', 'strains'])
        atomistic = Atomistic()
        extensions = dict(
            atomistic=(atomistic, atomistic.propertyExtension.propertyTable),
        )
        # with open(GATHEREDPATH/f'input/system{ID}', 'rt') as f:
        #     system = {'ID': ID, 'structure': Crystal.fromJSON(f.read())}

        structure = AtomicStructureRepresentation.readPOSCAR(GATHEREDPATH/f'input/system{ID}.vasp', (1, 1, 1))
        disassembler = Atomistic.atomicDisassemblerType(np.arange(len(structure)).reshape((-1, 1)))
        intermediate = disassembler.disassemble(structure)
        intermediate['.externalPressure'] = 100
        intermediate['atomistic.disassembler'] = disassembler
        intermediate = Flavour(extensions=extensions, **intermediate)
        result = interface.readOutput(intermediate, calcFolder=HOMEPATH/'gulp_test')
        self.assertTrue(np.isclose(result['.enthalpy'], -645.80329121))
        stress_ref = np.array([[-99.960848, 0.394719, -0.211383], [0.394719,  -100.008335,  0.019149], [-0.211383,  0.019149,  -100.271584]])
        self.assertTrue(np.allclose(result['.stressTensor'], stress_ref))
        strains_ref = np.array([0.012703, -0.031870, -0.039885, -0.000385, -0.008334, 0.182955])
        self.assertTrue(np.allclose(result['.strains'], strains_ref))

    def test_read_energy(self):
         with open(HOMEPATH/'gulp_test'/'output_bad_1st_SCF', 'rt') as f:
             contents = f.readlines()
         ev = GULP_Interface.readEnergy(None, contents)
         self.assertAlmostEqual(ev, -1604.1694, delta=0.001)
