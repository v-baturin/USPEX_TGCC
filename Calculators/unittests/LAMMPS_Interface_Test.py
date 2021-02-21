
import numpy as np
import os
import shutil
import unittest
import filecmp

from os.path import join as pj

from ...Atomistic.Crystal import Crystal

from ..LAMMPS_Interface import LAMMPS_Interface


HOMEPATH = os.path.dirname(os.path.abspath(__file__))
SPECIFICPATH = pj(HOMEPATH, 'lammpsSpecific')
GATHEREDPATH = pj(HOMEPATH, 'lammpsGatheredData')
WORKPATH = pj(HOMEPATH, 'C_lammps')


class LAMMPS_CalculatorTest(unittest.TestCase):


    def test_life(self):
        config = {'externalPressure' : 100}

        lammps = LAMMPS_Interface(tag='0', perturbate=False,
                                  libs=[pj(SPECIFICPATH, 'SiC.tersoff')], lammps_in=pj(SPECIFICPATH, 'lammps.in_1'))

        for ID in range(10):
            with open(pj(GATHEREDPATH, f'input/system{ID}'), 'rt') as f:
                system = {'ID': ID, 'structure': Crystal.fromJSON(f.read())}
            system['structure'].config = config
            os.mkdir(WORKPATH)
            lammps.prepareLocalCalculation(system, WORKPATH)
            folder = pj(GATHEREDPATH, 'input', f"CalcFold{system['ID']}")
            dcmp = filecmp.dircmp(folder, WORKPATH)
            match = not dcmp.diff_files
            for common_dir in dcmp.common_dirs:
                match = match and not dcmp.subdirs[common_dir].diff_files
            shutil.rmtree(WORKPATH)
            self.assertTrue(match)
            folder = pj(GATHEREDPATH, 'output')
            shutil.copytree(pj(folder, f"CalcFold{system['ID']}"), WORKPATH)
            lammps.readOutput(system, WORKPATH)
            shutil.rmtree(WORKPATH)
            system['structure']._fingerprint = None
            with open(pj(folder, f"system{system['ID']}"), 'rt') as f:
                systemRef = system['structure'].fromJSON(f.read())
            self.assertEqual(system['structure'], systemRef)


class LAMMPS_InterfaceTest(unittest.TestCase):
    def test_read_output(self):
        ID = 0
        # HERE what is written in ginput and goption no make sense.
        # Only output will be parsed and properties checked
        interface = LAMMPS_Interface(tag='0', perturbate=False,
                                  libs=[pj(SPECIFICPATH, 'SiC.tersoff')], lammps_in=pj(SPECIFICPATH, 'lammps.in_1'))
        with open(pj(GATHEREDPATH, f'input/system{ID}'), 'rt') as f:
            system = {'ID': ID, 'structure': Crystal.fromJSON(f.read())}

        interface.readOutput(system=system, calcFolder=pj(GATHEREDPATH, f'output/CalcFold{ID}'))
        self.assertTrue(np.isclose(system['enthalpy'], -105.503))

