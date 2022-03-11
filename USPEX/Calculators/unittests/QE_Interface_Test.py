import os
import shutil
import unittest
import filecmp

from os.path import join as pj


from ...Atomistic.RadialDistributionUtility import RadialDistributionUtility
from ...components import AtomisticRepresentation
from ..QE_Interface import QE_Interface


HOMEPATH = os.path.dirname(os.path.abspath(__file__))
SPECIFICPATH = pj(HOMEPATH, 'qeSpecific')
GATHEREDPATH = pj(HOMEPATH, 'qeGatheredData')
WORKPATH = pj(HOMEPATH, 'Ca4F8_qe')


class QE_CalculatorTest2(unittest.TestCase):
    """
    Checking correct parsing properties
    """
    def test_life(self):
        qe = QE_Interface(tag='1', options=pj(SPECIFICPATH, 'qEspresso_options_1'),
                          libs=[pj(SPECIFICPATH, 'SiC.C.pbe-van_bm.upf')], kresol=0.16)
        radialDistributionUtility = RadialDistributionUtility()


        for ID in range(10):
            with open(pj(GATHEREDPATH, f'input/system{ID}.vasp'), 'rt') as f:
                system = AtomisticRepresentation.readAtomicStructure(f)
                system['ID'] = ID
                system['externalPressure'] = 0.0001
            os.mkdir(WORKPATH)
            qe.prepareLocalCalculation(system, WORKPATH)
            folder = pj(GATHEREDPATH, 'input', f"CalcFold{system['ID']}")
            dcmp = filecmp.dircmp(folder, WORKPATH)
            match = not dcmp.diff_files
            for common_dir in dcmp.common_dirs:
                match = match and not dcmp.subdirs[common_dir].diff_files
            shutil.rmtree(WORKPATH)
            self.assertTrue(match)
            folder = pj(GATHEREDPATH, 'output')
            shutil.copytree(pj(folder, f"CalcFold{system['ID']}"), WORKPATH)
            qe.readOutput(system, WORKPATH)
            shutil.rmtree(WORKPATH)
            with open(pj(folder, f"system{system['ID']}.vasp"), 'rt') as f:
                systemRef = AtomisticRepresentation.readAtomicStructure(f)
            self.assertTrue(radialDistributionUtility.equal(system, systemRef))
