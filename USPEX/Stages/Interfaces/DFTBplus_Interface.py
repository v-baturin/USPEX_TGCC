"""
USPEX.Stages.DFTBplus_Interface
===========================

"""

import logging
import os
import numpy as np

from ase import Atoms
from ase.io.gen import read_gen, write_gen
from pathlib import Path

from .KPoints import KPoints, BadKPoints

logger = logging.getLogger(__name__)

HARTREE_TO_EV = 27.211386245988 #https://physics.nist.gov/cgi-bin/cuu/Value?hrev
GPA_TO_AU = 1.0/29421.015697
ANGSTROM_TO_BOHR = 1.0/0.529177210903


class DFTBplus_Interface:
    """
    Calculator for dftb+.
    Local running
    """
    DEFAULT_SLEEP_TIME = 30
    structureType = None
    atomType = None
    cellType = None

    inputFile, outputFile, errorFile = 'dftb_in.hsd', 'output', 'error'
    geometry_file = 'uspex.gen'
    specific_file = 'dftb_in.hsd_'
    kpoints_file = 'kpoints.uspex'
    pressure_file = 'pressure.uspex'
    movedAtoms_file = 'movedatoms.uspex'

    out_geometry_file = 'geo_end.gen'

    @classmethod
    def registerTypes(cls, structureType, atomType, cellType):
        cls.structureType = structureType
        cls.atomType = atomType
        cls.cellType = cellType

    def __init__(self, tag: str,
                       kresol: float = None,
                       dftb_input: str = None,
                       targetProperties: list = None,
                       **kwargs):

        self.tag = tag
        if dftb_input is None:
            dftb_input = Path.cwd()/f'Specific/{self.specific_file}{tag}'

        assert dftb_input.exists(), f'Please, check path to DFTB input. Now it is {dftb_input}'

        with open(dftb_input, 'r') as f:
            self.dftb_input = f.read()

        self.kPoints = KPoints(kresol) if kresol is not None else None
        self.targetProperties = targetProperties if targetProperties is not None else ['structure', 'enthalpy']

    def prepareLocalCalculation(self, system, calcFolder: Path):

        structure = system['structure']
        cell = structure.getCell()
        system['pbc'] = cell.getPBC()
        symbols = np.asarray([el.short_name for el in structure.getAtomTypes()])
        coordinates = structure.getCartesianCoordinates()
        cell_vectors = cell.getCellVectors()
        ase_struct = Atoms(symbols, positions=coordinates, cell=cell_vectors, pbc=system['pbc'])
        write_gen(calcFolder/self.geometry_file, ase_struct)

        if self.kPoints is None or cell.dim == 0:
            with open(calcFolder/self.kpoints_file, 'wt') as f:
                f.write('')
        else:
            try:
                kPoints = self.kPoints.build(cell)
            except BadKPoints:
                # This LATTICE is extremely wrong, let's skip it from now
                logger.info('K-points cannot be built, so it\'s set as   [1, 1, 1]')
                kPoints = [1, 1, 1]
            with open(calcFolder/self.kpoints_file, 'wt') as f:
                f.write('KPointsAndWeights = SupercellFolding {\n')
                f.write(f'{kPoints[0]}  0  0\n')
                f.write(f'0  {kPoints[1]}  0\n')
                f.write(f'0  0  {kPoints[2]}\n')
                f.write('0.0 0.0 0.0\n}\n')

        with open(calcFolder/self.pressure_file, 'wt') as f:
            if system['externalPressure']:
                f.write(f"Pressure [Pa] = {system['externalPressure']*10.0**9:10f}\n")
            else:
                f.write("")

        fixedIndices = system['disassembler'].envIndices[
            system['environment'].getFixedIndices()] if 'environment' in system else None
        if fixedIndices is not None:
            moved_atoms_string = 'MovedAtoms = !('
            onebased_fixedIndices = fixedIndices + 1
            indices = np.where(np.diff(onebased_fixedIndices) != 1)[0] + 1
            groups = np.split(onebased_fixedIndices, indices)
            for i, group in enumerate(groups):
                if len(group) == 1:
                    moved_atoms_string += f"{group[0]} "
                else:
                    moved_atoms_string += f"{group[0]}:{group[-1]} "
            moved_atoms_string += ')\n'
        else:
            moved_atoms_string = 'MovedAtoms = 1:-1\n'
        with open(calcFolder/self.movedAtoms_file, 'wt') as f:
            f.write(moved_atoms_string)

        with open(calcFolder/self.inputFile, 'wt') as dest:
            dest.write(self.dftb_input)

        return ''

    def isConverged(self, calcFolder: Path) -> bool:
        if not calcFolder.joinpath(self.outputFile).exists():
            return False

        with open(calcFolder/self.outputFile, 'rt') as f:
            content = f.read()
        if 'DFTB+ running times' not in content:
            logger.error('dftb+ is not completely Done')
            return False
        return True

    def readOutput(self, system, calcFolder: Path):
        new_structure = self.readStructure(calcFolder, system.pop('pbc'))
        EnergyHa = self.readEnergyHa(calcFolder)

        results = {}
        if 'structure' in self.targetProperties:
            results['structure'] = new_structure
        if 'energy' in self.targetProperties:
            results['energy'] = EnergyHa * HARTREE_TO_EV
        if 'enthalpy' in self.targetProperties:
            if system['structure'].getCell().dim == 3:
                results['enthalpy'] = (EnergyHa + \
                                       new_structure.getCell().getVolume() * system['externalPressure'] * \
                                      ANGSTROM_TO_BOHR**3.0 * GPA_TO_AU) * HARTREE_TO_EV
            else:
                results['enthalpy'] = EnergyHa * HARTREE_TO_EV

        return results

    def readStructure(self, calcFolder: Path, pbc):
        ase_struct = read_gen(calcFolder/self.out_geometry_file)
        new_lattice = []
        positions = ase_struct.get_positions()
        tmp_lattice = ase_struct.cell[:].copy()
        for i, vec in enumerate(tmp_lattice):
            if pbc[i]:
                new_lattice.append([float(x) for x in vec])
        cell = self.cellType.initFromCellVectors(pbc, new_lattice)
        new_structure = self.structureType(ase_struct.get_chemical_symbols(), positions, cell=cell)

        return new_structure

    def readEnergyHa(self, calcFolder: Path) -> float:
        with open(calcFolder/self.outputFile, 'rt') as f:
            content = f.readlines()
            for i in content:
                if 'Total Energy:' in i:
                    EnergyHa = float(i.split()[2])
        return EnergyHa
