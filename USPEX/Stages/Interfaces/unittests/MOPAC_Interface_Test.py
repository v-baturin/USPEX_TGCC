
import numpy as np
import shutil
import unittest
import filecmp

from pathlib import Path

from ....Optimizers.PoolEntry import PoolEntry, EntryFlavour
from USPEX.components import AtomicStructureRepresentation, MOPAC_Interface, Atomistic

HOMEPATH = Path(__file__).parent
SPECIFICPATH = HOMEPATH/'mopacSpecific'
GATHEREDPATH = HOMEPATH/'mopacGatheredData'
WORKPATH = HOMEPATH/'Si7O14_mopac'


class MOPAC_CalculatorTest(unittest.TestCase):

    def test_life(self):
        mopac = MOPAC_Interface(tag='0', mop_input=SPECIFICPATH/'mop_1')
        atomistic = Atomistic()
        extensions = dict(
            atomistic=atomistic.propertyExtension(atomistic)
        )

        for ID in range(10):
            structure = AtomicStructureRepresentation.readPOSCAR(GATHEREDPATH/f'input/system{ID}.vasp', (0, 0, 0))
            disassembler = Atomistic.atomicDisassemblerType(np.arange(len(structure)).reshape((-1, 1)))
            system = PoolEntry(ID, EntryFlavour(extensions=extensions))
            system.setProperty('externalPressure', 0.0)
            system.setProperty('disassembler', disassembler, extension='atomistic', suffix='intermediate')
            system.setProperty('disassembler', disassembler, extension='atomistic', suffix='0')
            system.setProperty('structure', structure, extension='atomistic', suffix='intermediate')
            WORKPATH.mkdir(parents=True, exist_ok=True)
            mopac.prepareLocalCalculation(system, WORKPATH)
            folder = GATHEREDPATH/'input'/f"CalcFold{system['ID']}"
            dcmp = filecmp.dircmp(folder, WORKPATH)
            match = not dcmp.diff_files
            for common_dir in dcmp.common_dirs:
                match = match and not dcmp.subdirs[common_dir].diff_files
            self.assertTrue(match)
            shutil.rmtree(WORKPATH)
            folder = GATHEREDPATH/'output'
            shutil.copytree(folder/f"CalcFold{system['ID']}", WORKPATH)
            mopac.readOutput(system, WORKPATH)
            shutil.rmtree(WORKPATH)
            structureRef = AtomicStructureRepresentation.readPOSCAR(folder/f"system{system['ID']}.vasp", (0, 0, 0))
            structure = system.getProperty('structure', extension='atomistic', suffix='0')
            self.assertTrue(np.allclose(structure.getCartesianCoordinates(), structureRef.getCartesianCoordinates()))

