
import numpy as np
import os
import shutil
import unittest
import filecmp

from os.path import join as pj


from ase import Atoms
from ase.io import write

# from ..MOPAC_Interface import MOPAC_Interface
from ...Atomistic.RadialDistributionUtility import RadialDistributionUtility
from ...components import AtomisticRepresentation, MOPAC_Interface, CellUtility, Cell

HOMEPATH = os.path.dirname(os.path.abspath(__file__))
SPECIFICPATH = pj(HOMEPATH, 'mopacSpecific')
GATHEREDPATH = pj(HOMEPATH, 'mopacGatheredData')
WORKPATH = pj(HOMEPATH, 'Si7O14_mopac')


class MOPAC_CalculatorTest(unittest.TestCase):


    def test_life(self):
        mopac = MOPAC_Interface(tag='0', mop_input=pj(SPECIFICPATH, 'mop_1'))
        radialDistributionUtility = RadialDistributionUtility()

        for ID in range(10):
            with open(pj(GATHEREDPATH, f'input/system{ID}.vasp'), 'rt') as f:
                system = AtomisticRepresentation.readAtomicStructure(f)
                system['externalPressure'] = 0
                system['ID'] = ID
                system['cell'] = type(system['cell'])(system['cell'].getCellVectors(), (False, False, False))
            os.mkdir(WORKPATH)
            mopac.prepareLocalCalculation(system, WORKPATH)
            folder = pj(GATHEREDPATH, 'input', f"CalcFold{system['ID']}")
            dcmp = filecmp.dircmp(folder, WORKPATH)
            match = not dcmp.diff_files
            for common_dir in dcmp.common_dirs:
                match = match and not dcmp.subdirs[common_dir].diff_files
            self.assertTrue(match)
            shutil.rmtree(WORKPATH)
            folder = pj(GATHEREDPATH, 'output')
            shutil.copytree(pj(folder, f"CalcFold{system['ID']}"), WORKPATH)
            mopac.readOutput(system, WORKPATH)
            shutil.rmtree(WORKPATH)
            with open(pj(folder, f"system{system['ID']}.vasp"), 'rt') as f:
                systemRef = AtomisticRepresentation.readAtomicStructure(f)
                systemRef['cell'] = type(systemRef['cell'])(systemRef['cell'].getCellVectors(), (False, False, False))
            self.assertTrue(radialDistributionUtility.equal(system, systemRef))

