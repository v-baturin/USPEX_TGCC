
import numpy as np
import os
import shutil
import unittest
import filecmp

from os.path import join as pj

from ..MOPAC_Interface import MOPAC_Interface
from ...Atomistic.RadialDistributionUtility import RadialDistributionUtility
from ...components import CrystalRepresentation

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
                system = CrystalRepresentation.readAtomicStructure(f)
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
                systemRef = CrystalRepresentation.readAtomicStructure(f)
            self.assertTrue(radialDistributionUtility.equal(system, systemRef))


# class LAMMPS_InterfaceTest(unittest.TestCase):
#     def test_read_output(self):
#         ID = 0
#         # HERE what is written in ginput and goption no make sense.
#         # Only output will be parsed and properties checked
#         interface = LAMMPS_Interface(tag='0', perturbate=False,
#                                   libs=[pj(SPECIFICPATH, 'SiC.tersoff')], lammps_in=pj(SPECIFICPATH, 'lammps.in_1'))
#         with open(pj(GATHEREDPATH, f'input/system{ID}.vasp'), 'rt') as f:
#             system = CrystalRepresentation.readAtomicStructure(f)
#         system['ID'] = 0
#         s, d = type(system['molecules'][0]).assemble(**system)
#         system['disassembler'] = d
#         system['atomTypes'] = s.getAtomTypes()
#
#
#         interface.readOutput(system=system, calcFolder=pj(GATHEREDPATH, f'output/CalcFold{ID}'))
#         self.assertTrue(np.isclose(system['enthalpy'], -105.503))

