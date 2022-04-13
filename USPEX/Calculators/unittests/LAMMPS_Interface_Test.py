
import numpy as np
import os
import shutil
import unittest
import filecmp

from os.path import join as pj

from ..LAMMPS_Interface import LAMMPS_Interface
from ...Atomistic.RadialDistributionUtility import RadialDistributionUtility
from ...components import AtomisticRepresentation

HOMEPATH = os.path.dirname(os.path.abspath(__file__))
SPECIFICPATH = pj(HOMEPATH, 'lammpsSpecific')
GATHEREDPATH = pj(HOMEPATH, 'lammpsGatheredData')
WORKPATH = pj(HOMEPATH, 'C_lammps')


class LAMMPS_CalculatorTest(unittest.TestCase):


    def test_life(self):
        lammps = LAMMPS_Interface(tag='0', perturbate=False,
                                  libs=[pj(SPECIFICPATH, 'SiC.tersoff')], lammps_in=pj(SPECIFICPATH, 'lammps.in_1'),
                                  specorder=['C'])
        radialDistributionUtility = RadialDistributionUtility()

        for ID in range(10):
            with open(pj(GATHEREDPATH, f'input/system{ID}.vasp'), 'rt') as f:
                system = AtomisticRepresentation.readAtomicStructure(f)
            system['externalPressure'] = 100
            system['ID'] = ID
            os.mkdir(WORKPATH)
            lammps.prepareLocalCalculation(system, WORKPATH)
            folder = pj(GATHEREDPATH, 'input', f"CalcFold{system['ID']}")
            dcmp = filecmp.dircmp(folder, WORKPATH)
            match = not dcmp.diff_files
            for common_dir in dcmp.common_dirs:
                match = match and not dcmp.subdirs[common_dir].diff_files
            self.assertTrue(match)
            shutil.rmtree(WORKPATH)
            folder = pj(GATHEREDPATH, 'output')
            shutil.copytree(pj(folder, f"CalcFold{system['ID']}"), WORKPATH)
            lammps.readOutput(system, WORKPATH)
            shutil.rmtree(WORKPATH)
            with open(pj(folder, f"system{system['ID']}.vasp"), 'rt') as f:
                systemRef = AtomisticRepresentation.readAtomicStructure(f)
            self.assertTrue(radialDistributionUtility.equal(system, systemRef))


class LAMMPS_InterfaceTest(unittest.TestCase):
    def test_read_output(self):
        ID = 0
        # HERE what is written in ginput and goption no make sense.
        # Only output will be parsed and properties checked
        interface = LAMMPS_Interface(tag='0', perturbate=False,
                                  libs=[pj(SPECIFICPATH, 'SiC.tersoff')], lammps_in=pj(SPECIFICPATH, 'lammps.in_1'), specorder=['C'])
        with open(pj(GATHEREDPATH, f'input/system{ID}.vasp'), 'rt') as f:
            system = AtomisticRepresentation.readAtomicStructure(f)
        system['ID'] = 0
        s, d = type(system['molecules'][0]).assemble(**system)
        system['disassembler'] = d
        system['atomTypes'] = s.getAtomTypes()
        system['assembledCell'] = system['cell']

        interface.readOutput(system=system, calcFolder=pj(GATHEREDPATH, f'output/CalcFold{ID}'))
        self.assertTrue(np.isclose(system['enthalpy'], -102.64364))

