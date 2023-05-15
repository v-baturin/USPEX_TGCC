"""
USPEX.Stages.DFTBplus_Interface
===========================

"""

import logging
import os
from os.path import join as pj
import numpy as np

from ase import Atoms
from ase.io.gen import read_gen, write_gen

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
    fixedIndices_file = 'fixed.uspex'

    out_geometry_file = 'geo_end.gen'

    @classmethod
    def registerTypes(cls, structureType, atomType, cellType):
        cls.structureType = structureType
        cls.atomType = atomType
        cls.cellType = cellType

    def __init__(self, tag: str, kresol: float = None, dftb_input: str = None, targetProperties: list = None, **kwargs):

        self.tag = tag
        if dftb_input is None:
            dftb_input = pj(os.getcwd(), f'Specific/{self.specific_file}{tag}')

        assert os.path.exists(dftb_input)

        with open(dftb_input, 'r') as f:
            self.dftb_input = f.read()

        self.kPoints = KPoints(kresol) if kresol is not None else None
        self.targetProperties = targetProperties if targetProperties is not None else ['structure', 'enthalpy']

    def prepareLocalCalculation(self, system, calcFolder : str):

        structure = system['structure']
        cell = structure.getCell()
        system['pbc'] = cell.getPBC()
        symbols = np.asarray([el.short_name for el in structure.getAtomTypes()])
        coordinates = structure.getCartesianCoordinates()
        cell_vectors = cell.getCellVectors()
        ase_struct = Atoms(symbols, positions=coordinates, cell=cell_vectors, pbc=system['pbc'])
        write_gen(pj(calcFolder, self.geometry_file), ase_struct)

        if self.kPoints is not None:
            try:
                kPoints = self.kPoints.build(cell)
            except BadKPoints:
                # This LATTICE is extremely wrong, let's skip it from now
                logger.info('K-points cannot be built, so it\'s set as   [1, 1, 1]')
                kPoints = [1, 1, 1]
            with open(pj(calcFolder, self.kpoints_file), 'a') as f:
                f.write('{}  0  0\n'.format(kPoints[0]))
                f.write('0  {}  0\n'.format(kPoints[1]))
                f.write('0  0  {}\n'.format(kPoints[2]))
                f.write('0.0 0.0 0.0\n')

        if system['externalPressure']:
            with open(pj(calcFolder, self.pressure_file), 'a') as myfile:
                myfile.write(f"Pressure [Pa] = {system['externalPressure']*10.0**9:10f}\n")

        fixedIndices = system['disassembler'].envIndices[
            system['environment'].getFixedIndices()] if 'environment' in system else None
        if fixedIndices is not None:
            fixed_indices_string = 'MovedAtoms = !('
            onebased_fixedIndices = fixedIndices + 1
            indices = np.where(np.diff(onebased_fixedIndices) != 1)[0] + 1
            groups = np.split(onebased_fixedIndices, indices)
            for i, group in enumerate(groups):
                if len(group) == 1:
                    fixed_indices_string += f"{group[0]} "
                else:
                    fixed_indices_string += f"{group[0]}:{group[-1]} "
            fixed_indices_string += ')\n'
            with open(pj(calcFolder, self.fixedIndices_file), 'a') as f:
                f.write(fixed_indices_string)

        with open(pj(calcFolder, self.inputFile), 'wt') as dest:
            dest.write(self.dftb_input)

        return ''

    def isConverged(self, calcFolder: str):

        if not os.path.exists(pj(calcFolder, self.outputFile)):
            return False

        with open(pj(calcFolder, self.outputFile), 'rt') as f:
            content = f.read()
        if 'DFTB+ running times' not in content:
            logger.error('dftb+ is not completely Done')
            return False
        return True

    def readOutput(self, system, calcFolder: str):

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

    def readStructure(self, calcFolder: str, pbc):

        ase_struct = read_gen(pj(calcFolder, self.out_geometry_file))
        atomTypes = []
        new_lattice = []
        for i in ase_struct.get_chemical_symbols():
            atomTypes.append(self.atomType(i))
        positions = ase_struct.get_positions()
        tmp_lattice = ase_struct.cell[:].copy()
        for i, vec in enumerate(tmp_lattice):
            if pbc[i]:
                new_lattice.append([float(x) for x in vec])
        cell = self.cellType.initFromCellVectors(pbc, new_lattice)
        new_structure = self.structureType(atomTypes, positions, cell=cell)

        return new_structure

    def readEnergyHa(self, calcFolder: str):

        with open(pj(calcFolder, self.outputFile), 'rt') as f:
            content = f.readlines()
            for i in content:
                if 'Total Energy:' in i:
                    EnergyHa = float(i.split()[2])

        return EnergyHa