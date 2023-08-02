
import numpy as np
import shutil
import unittest
import filecmp

from pathlib import Path

from ....Optimizers.PoolEntry import PoolEntry
from ....components import AtomisticRepresentation, LAMMPS_Interface, Atomistic

HOMEPATH = Path(__file__).parent
SPECIFICPATH = HOMEPATH/'lammpsSpecific'
GATHEREDPATH = HOMEPATH/'lammpsGatheredData'
WORKPATH = HOMEPATH/'C_lammps'


class LAMMPS_CalculatorTest(unittest.TestCase):


    def test_life(self):
        lammps = LAMMPS_Interface(tag='0',
                                  libs=[SPECIFICPATH/'SiC.tersoff'], lammps_in=SPECIFICPATH/'lammps.in_1',
                                  specorder=['C'])
        atomistic = Atomistic()
        extensions = dict(
            atomistic=atomistic.propertyExtension(atomistic)
        )

        for ID in range(10):
            structure = AtomisticRepresentation.readPOSCAR(GATHEREDPATH/f'input/system{ID}.vasp', (1, 1, 1))
            disassembler = AtomisticRepresentation.atomicDisassemblerType(np.arange(len(structure)).reshape((-1, 1)))
            system = PoolEntry(extensions=extensions, ID=ID)
            system.setProperty('externalPressure', 100.0)
            system.setProperty('disassembler', disassembler, prefix='atomistic', suffix='intermediate')
            system.setProperty('disassembler', disassembler, prefix='atomistic', suffix='0')
            system.setProperty('structure', structure, prefix='atomistic', suffix='intermediate')

            WORKPATH.mkdir(parents=True, exist_ok=True)
            lammps.prepareLocalCalculation(system, WORKPATH)
            folder = GATHEREDPATH/'input'/f"CalcFold{system['ID']}"
            dcmp = filecmp.dircmp(folder, WORKPATH)
            match = not dcmp.diff_files
            for common_dir in dcmp.common_dirs:
                match = match and not dcmp.subdirs[common_dir].diff_files
            self.assertTrue(match)
            shutil.rmtree(WORKPATH)
            folder = GATHEREDPATH/'output'
            shutil.copytree(folder/f"CalcFold{system['ID']}", WORKPATH)
            lammps.readOutput(system, WORKPATH)
            shutil.rmtree(WORKPATH)
            structureRef = AtomisticRepresentation.readPOSCAR(folder/f"system{system['ID']}.vasp", (1, 1, 1))
            structure = system.getProperty('structure', prefix='atomistic', suffix='0')
            cell = structure.getCell()
            cellRef = structureRef.getCell()
            self.assertTrue(np.allclose(cell.getCellVectors(),
                                        cellRef.getCellVectors()))
            # self.assertTrue(np.allclose(cell.getWrapedCartesianCoordinates(results['structure'].getCartesianCoordinates()),
            #                             cellRef.getWrapedCartesianCoordinates(structureRef.getCartesianCoordinates())))


class LAMMPS_InterfaceTest(unittest.TestCase):
    def test_read_output(self):
        ID = 0
        # HERE what is written in ginput and goption no make sense.
        # Only output will be parsed and properties checked
        interface = LAMMPS_Interface(tag='0', libs=[SPECIFICPATH/'SiC.tersoff'],
                                     lammps_in=SPECIFICPATH/'lammps.in_1', specorder=['C'])
        atomistic = Atomistic()
        extensions = dict(
            atomistic=atomistic.propertyExtension(atomistic)
        )

        structure = AtomisticRepresentation.readPOSCAR(GATHEREDPATH/f'input/system{ID}.vasp', (1, 1, 1))
        disassembler = AtomisticRepresentation.atomicDisassemblerType(np.arange(len(structure)).reshape((-1, 1)))
        system = PoolEntry(extensions=extensions, ID=ID)
        system.setProperty('externalPressure', 0.0)
        system.setProperty('disassembler', disassembler, prefix='atomistic', suffix='intermediate')
        system.setProperty('disassembler', disassembler, prefix='atomistic', suffix='0')
        system.setProperty('structure', structure, prefix='atomistic', suffix='intermediate')

        interface.readOutput(system=system, calcFolder=GATHEREDPATH/f'output/CalcFold{ID}')
        self.assertTrue(np.isclose(system['.enthalpy.0'], -102.64364))


class LAMMPS_MLIP_Test(unittest.TestCase):

    def setUp(self) -> None:
        self.interface = LAMMPS_Interface(tag='0', lammps_in=HOMEPATH/'LAMMPS_MLIP_SAMPLE'/'lammps.in',
                                          mlip_in=HOMEPATH/'LAMMPS_MLIP_SAMPLE/mlip.ini',
                                          mlip=HOMEPATH/'LAMMPS_MLIP_SAMPLE/p.mtp',
                                          specorder=['Li', 'B', 'H'], targetProperties=['trajectory'])
        atomistic = Atomistic()
        self.extensions = dict(
            atomistic=atomistic.propertyExtension(atomistic)
        )

    def test_init(self):
        structure = AtomisticRepresentation.readPOSCAR(HOMEPATH/'LAMMPS_MLIP_SAMPLE/Li_B_H_POSCAR', (1, 1, 1))
        disassembler = AtomisticRepresentation.atomicDisassemblerType(np.arange(len(structure)).reshape((-1, 1)))
        system = PoolEntry(extensions=self.extensions, ID=0)
        system.setProperty('disassembler', disassembler, prefix='atomistic', suffix='intermediate')
        system.setProperty('disassembler', disassembler, prefix='atomistic', suffix='0')
        system.setProperty('structure', structure, prefix='atomistic', suffix='intermediate')
        calcFolder = HOMEPATH/'LAMMPS_MLIP_INIT'
        calcFolder.mkdir()
        self.interface.prepareLocalCalculation(system=system, calcFolder=calcFolder)
        refFolder = HOMEPATH/'LAMMPS_MLIP_REF'
        dcmp = filecmp.dircmp(refFolder, calcFolder)
        match = not dcmp.diff_files
        for common_dir in dcmp.common_dirs:
            match = match and not dcmp.subdirs[common_dir].diff_files
        self.assertTrue(match)
        shutil.rmtree(calcFolder)


    def test_sample(self):
        system = PoolEntry(extensions=self.extensions, ID=0)
        self.interface.readOutput(system=system, calcFolder=HOMEPATH/'LAMMPS_MLIP_SAMPLE')
        self.assertEqual(len(system['.trajectory.0']), 1805)
        for system in system['.trajectory.0']:
            self.assertEqual(len(system['structure']), 104)
