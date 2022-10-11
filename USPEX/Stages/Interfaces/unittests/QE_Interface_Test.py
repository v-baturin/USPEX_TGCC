import filecmp
import shutil
import unittest

from pathlib import Path


from USPEX.Atomistic.RadialDistributionUtility import RadialDistributionUtility
from USPEX.components import AtomisticRepresentation
from ..QE_Interface import QE_Interface


HOMEPATH = Path(__file__).parent
SPECIFICPATH = HOMEPATH/'qeSpecific'
GATHEREDPATH = HOMEPATH/'qeGatheredData'
WORKPATH = HOMEPATH/'Ca4F8_qe'


class QE_CalculatorTest2(unittest.TestCase):
    """
    Checking correct parsing properties
    """
    def test_life(self):
        qe = QE_Interface(tag='1',
                          options=SPECIFICPATH/'qEspresso_options_1',
                          pseudopotentials={'C': SPECIFICPATH/'C.pbe-van_bm.upf'},
                          kresol=0.16)
        radialDistributionUtility = RadialDistributionUtility(symbols=['C'])

        for ID in range(10):
            system = AtomisticRepresentation.readAtomicStructure(GATHEREDPATH/f'input/system{ID}.vasp')
            system['ID'] = ID
            system['externalPressure'] = 0.0001
            system['tmp_0'] = {}
            WORKPATH.mkdir(exist_ok=True)
            qe.prepareLocalCalculation(system, WORKPATH)
            folder = GATHEREDPATH/'input'/f"CalcFold{system['ID']}"
            dcmp = filecmp.dircmp(folder, WORKPATH)
            match = not dcmp.diff_files
            for common_dir in dcmp.common_dirs:
                match = match and not dcmp.subdirs[common_dir].diff_files
            shutil.rmtree(WORKPATH)
            self.assertTrue(match)
            folder = GATHEREDPATH/'output'
            shutil.copytree(folder/f"CalcFold{system['ID']}", WORKPATH)
            qe.readOutput(system, WORKPATH)
            shutil.rmtree(WORKPATH)
            systemRef = AtomisticRepresentation.readAtomicStructure(folder/f"system{system['ID']}.vasp")
            self.assertTrue(radialDistributionUtility.equal(system, systemRef))
