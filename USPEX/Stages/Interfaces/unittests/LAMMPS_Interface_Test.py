
import numpy as np
import shutil
import unittest
import filecmp

from pathlib import Path

from ..LAMMPS_Interface import LAMMPS_Interface
from ....Optimizers.PoolEntry import EntryFlavour
from ....Atomistic.Primitives.Element import Element
from ....Atomistic.Primitives.Cell import Cell
from ....Atomistic.Primitives.AtomicStructure import AtomicStructure
from ....IO.AtomicStructureRepresentation import AtomicStructureRepresentation
from ....Atomistic.Atomistic import Atomistic
Atomistic.registerTypes(AtomicStructure, Element, Cell, AtomicStructureRepresentation)
LAMMPS_Interface.registerTypes(AtomicStructureRepresentation)


HOMEPATH = Path(__file__).parent
SPECIFICPATH = HOMEPATH/'lammpsSpecific'
GATHEREDPATH = HOMEPATH/'lammpsGatheredData'
WORKPATH = HOMEPATH/'C_lammps'


class LAMMPS_CalculatorTest(unittest.TestCase):


    def test_life(self):
        lammps = LAMMPS_Interface(tag='0',
                                  libs=[SPECIFICPATH/'SiC.tersoff'], lammps_in=SPECIFICPATH/'lammps.in_1',
                                  specorder=['C'],
                                  targetProperties=['structure', 'enthalpy'])
        atomistic = Atomistic()
        extensions = dict(
            atomistic=atomistic.propertyExtension(atomistic)
        )

        for ID in range(10):
            structure = AtomicStructureRepresentation.readPOSCAR(GATHEREDPATH/f'input/system{ID}.vasp', (1, 1, 1))
            disassembler = Atomistic.atomicDisassemblerType(np.arange(len(structure)).reshape((-1, 1)))
            intermediate = dict()
            intermediate['atomistic.structure'] = structure
            intermediate['atomistic.disassembler'] = disassembler
            intermediate['.externalPressure'] = 100.0
            intermediate['.ID'] = ID
            intermediate = EntryFlavour(extensions=extensions, **intermediate)
            WORKPATH.mkdir(parents=True, exist_ok=True)
            lammps.prepareLocalCalculation(intermediate, WORKPATH)
            folder = GATHEREDPATH/'input'/f"CalcFold{ID}"
            dcmp = filecmp.dircmp(folder, WORKPATH)
            match = not dcmp.diff_files
            for common_dir in dcmp.common_dirs:
                match = match and not dcmp.subdirs[common_dir].diff_files
            self.assertTrue(match)
            shutil.rmtree(WORKPATH)
            folder = GATHEREDPATH/'output'
            shutil.copytree(folder/f"CalcFold{ID}", WORKPATH)
            result = lammps.readOutput(intermediate, WORKPATH)
            shutil.rmtree(WORKPATH)
            structureRef = AtomicStructureRepresentation.readPOSCAR(folder/f"system{ID}.vasp", (1, 1, 1))
            structure = result.getProperty('structure', extension='atomistic')
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
                                     lammps_in=SPECIFICPATH/'lammps.in_1', specorder=['C'], targetProperties=['structure', 'enthalpy'])
        atomistic = Atomistic()
        extensions = dict(
            atomistic=atomistic.propertyExtension(atomistic)
        )

        structure = AtomicStructureRepresentation.readPOSCAR(GATHEREDPATH/f'input/system{ID}.vasp', (1, 1, 1))
        disassembler = Atomistic.atomicDisassemblerType(np.arange(len(structure)).reshape((-1, 1)))
        intermediate = disassembler.disassemble(structure)
        intermediate['.externalPressure'] = 0.0
        intermediate['.ID'] = ID
        intermediate['atomistic.disassembler'] = disassembler
        intermediate = EntryFlavour(extensions=extensions, **intermediate)
        result = interface.readOutput(intermediate, calcFolder=GATHEREDPATH/f'output/CalcFold{ID}')
        self.assertTrue(np.isclose(result['.enthalpy'], -102.64364))


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
        structure = AtomicStructureRepresentation.readPOSCAR(HOMEPATH/'LAMMPS_MLIP_SAMPLE/Li_B_H_POSCAR', (1, 1, 1))
        disassembler = Atomistic.atomicDisassemblerType(np.arange(len(structure)).reshape((-1, 1)))
        intermediate = dict()
        intermediate['.externalPressure'] = 0.0
        intermediate['.ID'] = 0
        intermediate['atomistic.structure'] = structure
        intermediate['atomistic.disassembler'] = disassembler
        intermediate = EntryFlavour(extensions=self.extensions, **intermediate)
        calcFolder = HOMEPATH/'LAMMPS_MLIP_INIT'
        calcFolder.mkdir()
        self.interface.prepareLocalCalculation(intermediate, calcFolder=calcFolder)
        refFolder = HOMEPATH/'LAMMPS_MLIP_REF'
        dcmp = filecmp.dircmp(refFolder, calcFolder)
        match = not dcmp.diff_files
        for common_dir in dcmp.common_dirs:
            match = match and not dcmp.subdirs[common_dir].diff_files
        self.assertTrue(match)
        shutil.rmtree(calcFolder)


    def test_sample(self):
        result = self.interface.readOutput(EntryFlavour(extensions=self.extensions), calcFolder=HOMEPATH/'LAMMPS_MLIP_SAMPLE')
        self.assertEqual(len(result['.trajectory']), 1805)
        for system in result['.trajectory']:
            self.assertEqual(len(system['structure']), 104)
