
import numpy as np
import shutil
import unittest
import filecmp

from pathlib import Path

from USPEX.components import AtomisticRepresentation, MOPAC_Interface

HOMEPATH = Path(__file__).parent
SPECIFICPATH = HOMEPATH/'mopacSpecific'
GATHEREDPATH = HOMEPATH/'mopacGatheredData'
WORKPATH = HOMEPATH/'Si7O14_mopac'


class MOPAC_CalculatorTest(unittest.TestCase):

    def test_life(self):
        mopac = MOPAC_Interface(tag='0', mop_input=pj(SPECIFICPATH, 'mop_1'))

        for ID in range(10):
            structure = AtomisticRepresentation.readPOSCAR(pj(GATHEREDPATH, f'input/system{ID}.vasp'), (0, 0, 0))
            system = dict(
                ID=ID,
                structure=structure,
                disassembler=AtomisticRepresentation.atomicDisassemblerType(
                    np.arange(len(structure)).reshape((-1, 1))),
                externalPressure=0.0
            )
            os.mkdir(WORKPATH)
            mopac.prepareLocalCalculation(system, WORKPATH)
            folder = GATHEREDPATH/'input'/f"CalcFold{system['ID']}"
            dcmp = filecmp.dircmp(folder, WORKPATH)
            match = not dcmp.diff_files
            for common_dir in dcmp.common_dirs:
                match = match and not dcmp.subdirs[common_dir].diff_files
            self.assertTrue(match)
            shutil.rmtree(WORKPATH)
            folder = pj(GATHEREDPATH, 'output')
            shutil.copytree(pj(folder, f"CalcFold{system['ID']}"), WORKPATH)
            results = mopac.readOutput(system, WORKPATH)
            shutil.rmtree(WORKPATH)
            structureRef = AtomisticRepresentation.readPOSCAR(pj(folder, f"system{system['ID']}.vasp"), (0, 0, 0))
            self.assertTrue(np.allclose(results['structure'].getCartesianCoordinates(),
                                        structureRef.getCartesianCoordinates()))

