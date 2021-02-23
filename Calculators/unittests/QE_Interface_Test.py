import os
import shutil
import unittest
import filecmp

from os.path import join as pj


from ...Atomistic.Crystal import Crystal
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

        for ID in range(10):
            with open(pj(GATHEREDPATH, f'input/system{ID}'), 'rt') as f:
                system = {'ID': ID, 'structure': Crystal.fromJSON(f.read())}
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
            system['structure']._fingerprint = None
            with open(pj(folder, f"system{system['ID']}"), 'rt') as f:
                systemRef = system['structure'].fromJSON(f.read())
            self.assertEqual(system['structure'], systemRef)
