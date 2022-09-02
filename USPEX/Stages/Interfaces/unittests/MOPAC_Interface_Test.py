
import numpy as np
import shutil
import unittest
import filecmp

from pathlib import Path


from ase import Atoms
from ase.io import write

# from ..MOPAC_Interface import MOPAC_Interface
from USPEX.Atomistic.RadialDistributionUtility import RadialDistributionUtility
from USPEX.components import AtomisticRepresentation, MOPAC_Interface, CellUtility, Cell

HOMEPATH = Path(__file__).parent
SPECIFICPATH = HOMEPATH/'mopacSpecific'
GATHEREDPATH = HOMEPATH/'mopacGatheredData'
WORKPATH = HOMEPATH/'Si7O14_mopac'


class MOPAC_CalculatorTest(unittest.TestCase):

    def test_life(self):
        mopac = MOPAC_Interface(tag='0', mop_input=SPECIFICPATH/'mop_1')
        radialDistributionUtility = RadialDistributionUtility(symbols=['Si', 'O'])

        for ID in range(10):
            with open(GATHEREDPATH/f'input/system{ID}.vasp', 'rt') as f:
                system = AtomisticRepresentation.readAtomicStructure(f)
                system['externalPressure'] = 0
                system['ID'] = ID
                system['cell'] = type(system['cell'])(system['cell'].getCellVectors(), (False, False, False))
            WORKPATH.mkdir(exist_ok=True)
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
            with open(folder/f"system{system['ID']}.vasp", 'rt') as f:
                systemRef = AtomisticRepresentation.readAtomicStructure(f)
                systemRef['cell'] = type(systemRef['cell'])(systemRef['cell'].getCellVectors(), (False, False, False))
            self.assertTrue(radialDistributionUtility.equal(system, systemRef))
