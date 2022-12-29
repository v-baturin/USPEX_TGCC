
import numpy as np
import os
import shutil
import unittest
import filecmp

from os.path import join as pj

from ....components import AtomisticRepresentation, LAMMPS_Interface

HOMEPATH = os.path.dirname(os.path.abspath(__file__))
SPECIFICPATH = pj(HOMEPATH, 'lammpsSpecific')
GATHEREDPATH = pj(HOMEPATH, 'lammpsGatheredData')
WORKPATH = pj(HOMEPATH, 'C_lammps')


class LAMMPS_CalculatorTest(unittest.TestCase):


    def test_life(self):
        lammps = LAMMPS_Interface(tag='0',
                                  libs=[pj(SPECIFICPATH, 'SiC.tersoff')], lammps_in=pj(SPECIFICPATH, 'lammps.in_1'),
                                  specorder=['C'])

        for ID in range(10):
            structure = AtomisticRepresentation.readPOSCAR(pj(GATHEREDPATH, f'input/system{ID}.vasp'), (1, 1, 1))
            system = dict(
                ID=ID,
                structure=structure,
                disassembler=AtomisticRepresentation.atomicDisassemblerType(
                    np.arange(len(structure)).reshape((-1, 1))),
                externalPressure=100.0
            )
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
            results = lammps.readOutput(system, WORKPATH)
            shutil.rmtree(WORKPATH)
            structureRef = AtomisticRepresentation.readPOSCAR(pj(folder, f"system{system['ID']}.vasp"), (1, 1, 1))
            cell = results['structure'].getCell()
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
        interface = LAMMPS_Interface(tag='0', libs=[pj(SPECIFICPATH, 'SiC.tersoff')],
                                     lammps_in=pj(SPECIFICPATH, 'lammps.in_1'), specorder=['C'])
        structure = AtomisticRepresentation.readPOSCAR(pj(GATHEREDPATH, f'input/system{ID}.vasp'), (1, 1, 1))
        system = dict(
            ID=ID,
            structure=structure,
            disassembler=AtomisticRepresentation.atomicDisassemblerType(
                np.arange(len(structure)).reshape((-1, 1))),
            ase={'pbc': (1, 1, 1)},
            externalPressure=0.0
        )
        results = interface.readOutput(system=system, calcFolder=pj(GATHEREDPATH, f'output/CalcFold{ID}'))
        self.assertTrue(np.isclose(results['enthalpy'], -102.64364))


class LAMMPS_MLIP_Test(unittest.TestCase):

    def setUp(self) -> None:
        self.interface = LAMMPS_Interface(tag='0', lammps_in=pj(HOMEPATH, 'LAMMPS_MLIP_SAMPLE', 'lammps.in'),
                                          mlip_in=pj(HOMEPATH, 'LAMMPS_MLIP_SAMPLE', 'mlip.ini'),
                                          mlip=pj(HOMEPATH, 'LAMMPS_MLIP_SAMPLE', 'p.mtp'),
                                          specorder=['Li', 'B', 'H'], targetProperties=['trajectory'])

    def test_init(self):
        structure = AtomisticRepresentation.readPOSCAR(pj(HOMEPATH, 'LAMMPS_MLIP_SAMPLE', 'Li_B_H_POSCAR'), (1, 1, 1))
        system = dict(
            ID=0,
            structure=structure,
            disassembler=AtomisticRepresentation.atomicDisassemblerType(
                np.arange(len(structure)).reshape((-1, 1))),
        )
        calcFolder = pj(HOMEPATH, 'LAMMPS_MLIP_INIT')
        os.mkdir(calcFolder)
        self.interface.prepareLocalCalculation(system=system, calcFolder=calcFolder)
        refFolder = pj(HOMEPATH, 'LAMMPS_MLIP_REF')
        dcmp = filecmp.dircmp(refFolder, calcFolder)
        match = not dcmp.diff_files
        for common_dir in dcmp.common_dirs:
            match = match and not dcmp.subdirs[common_dir].diff_files
        self.assertTrue(match)
        shutil.rmtree(calcFolder)


    def test_sample(self):
        system = dict(
            ase={'pbc': (1, 1, 1)},
            disassembler=None,
            externalPressure=0.0
        )

        results = self.interface.readOutput(system=system, calcFolder=pj(HOMEPATH, 'LAMMPS_MLIP_SAMPLE'))
        self.assertEqual(len(results['trajectory']), 1805)
        for system in results['trajectory']:
            self.assertEqual(len(system['structure']), 104)
