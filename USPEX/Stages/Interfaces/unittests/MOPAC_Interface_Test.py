
import numpy as np
import shutil
import unittest
import filecmp

from pathlib import Path

from ..MOPAC_Interface import MOPAC_Interface
from ....DataModel.Flavour import Flavour
from ....Atomistic.Primitives.Element import Element
from ....Atomistic.Primitives.Cell import Cell
from ....Atomistic.Primitives.AtomicStructure import AtomicStructure
from ....IO.AtomicStructureRepresentation import AtomicStructureRepresentation
from ....Atomistic.Atomistic import Atomistic
Atomistic.registerTypes(AtomicStructure, Element, Cell, AtomicStructureRepresentation)


HOMEPATH = Path(__file__).parent
SPECIFICPATH = HOMEPATH/'mopacSpecific'
GATHEREDPATH = HOMEPATH/'mopacGatheredData'
WORKPATH = HOMEPATH/'Si7O14_mopac'


class MOPAC_CalculatorTest(unittest.TestCase):

    def test_life(self):
        mopac = MOPAC_Interface(tag='0', mop_input=SPECIFICPATH/'mop_1', targetProperties=['structure', 'enthalpy'])
        atomistic = Atomistic()
        extensions = dict(
            atomistic=(atomistic, atomistic.propertyExtension.propertyTable),
        )

        for ID in range(10):
            structure = AtomicStructureRepresentation.readPOSCAR(GATHEREDPATH/f'input/system{ID}.vasp', (0, 0, 0))
            disassembler = Atomistic.atomicDisassemblerType(np.arange(len(structure)).reshape((-1, 1)))
            intermediate = dict()
            intermediate['atomistic.structure'] = structure
            intermediate['.externalPressure'] = 0.0
            intermediate['atomistic.disassembler'] = disassembler
            intermediate = Flavour(extensions=extensions, **intermediate)
            WORKPATH.mkdir(parents=True, exist_ok=True)
            mopac.prepareLocalCalculation(intermediate, WORKPATH)
            folder = GATHEREDPATH/'input'/f"CalcFold{ID}"
            dcmp = filecmp.dircmp(folder, WORKPATH)
            match = not dcmp.diff_files
            for common_dir in dcmp.common_dirs:
                match = match and not dcmp.subdirs[common_dir].diff_files
            self.assertTrue(match)
            shutil.rmtree(WORKPATH)
            folder = GATHEREDPATH/'output'
            shutil.copytree(folder/f"CalcFold{ID}", WORKPATH)
            result = mopac.readOutput(intermediate, WORKPATH)
            shutil.rmtree(WORKPATH)
            structureRef = AtomicStructureRepresentation.readPOSCAR(folder/f"system{ID}.vasp", (0, 0, 0))
            structure = result.getProperty('structure', extension='atomistic')
            self.assertTrue(np.allclose(structure.getCartesianCoordinates(), structureRef.getCartesianCoordinates()))

