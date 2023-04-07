"""
USPEX.Stages.CP2K_Interface
===========================

"""

import logging
import os
import shutil
import numpy as np
from os.path import join as pj
from typing import List

from ase.atoms import Atoms
from ase.io import read, write

from .KPoints import KPoints, BadKPoints

logger = logging.getLogger(__name__)
HARTREE_TO_EV = 27.211386245988 #https://physics.nist.gov/cgi-bin/cuu/Value?hrev
GPA_TO_AU = 1.0/29421.015697
ANGSTROM_TO_BOHR = 1.0/0.529177210903

class CP2K_Interface:
    """
    Calculator for CP2K.
    Local running
    """
    DEFAULT_SLEEP_TIME = 30
    structureType = None
    atomType = None
    cellType = None

    inputFile, outputFile, errorFile = 'cp2k.inp', 'output', 'error'
    cell_file = 'cell.uspex'
    geometry_file = 'geometry.uspex'
    kpoints_file = 'kpoints.uspex'
    pressure_file = 'pressure.uspex'
    fixedIndices_file = 'fixed.uspex'
    atomIndices_file = '_list.uspex'
    specific_file = 'cp2k_in_'

    out_geometry_file = 'USPEX-pos-1.xyz'
    out_cell_file = 'USPEX-1.cell'

    @classmethod
    def registerTypes(cls, structureType, atomType, cellType):
        cls.structureType = structureType
        cls.atomType = atomType
        cls.cellType = cellType

    def __init__(self, tag: str, kresol: float = None, cp2k_in: str = None, fixCell: bool = False,
                 targetProperties: list = None, **kwargs):

        self.tag = tag
        if cp2k_in is None:
            cp2k_in = pj(os.getcwd(), f'Specific/{self.specific_file}{tag}')

        assert os.path.exists(cp2k_in)

        with open(cp2k_in, 'r') as f:
            self.cp2k_in = f.read()

        self.kPoints = KPoints(kresol) if kresol is not None else None

        self.fixCell = fixCell
        self.targetProperties = targetProperties if targetProperties is not None else ['structure', 'enthalpy']

    def prepareLocalCalculation(self, system, calcFolder : str):
        structure = system['structure']

        cell = structure.getCell()
        system['pbc'] = cell.getPBC()

        with open(pj(calcFolder, self.inputFile), 'wt') as dest:
            dest.write(self.cp2k_in)

        if self.kPoints is not None:
            try:
                kPoints = self.kPoints.build(cell)
            except BadKPoints:
                # This LATTICE is extremely wrong, let's skip it from now
                logger.info('K-points cannot be built, so it\'s set as   [1, 1, 1]')
                kPoints = [1, 1, 1]
            with open(pj(calcFolder, self.kpoints_file), 'a') as f:
                f.write('SCHEME MONKHORST-PACK {} {} {}'.format(*kPoints))

        with open(pj(calcFolder, self.cell_file), 'wt') as fp:
            lat = cell.getCellVectors()
            fp.write('A   {:12.6f} {:12.6f} {:12.6f}\n'.format(*lat[0, :]))
            fp.write('B   {:12.6f} {:12.6f} {:12.6f}\n'.format(*lat[1, :]))
            fp.write('C   {:12.6f} {:12.6f} {:12.6f}\n'.format(*lat[2, :]))

        with open(pj(calcFolder, self.geometry_file), 'wt') as fp:
            for i, (symbol, coord) in enumerate(zip(structure.getAtomTypes(), structure.getCartesianCoordinates())):
                fp.write('{0:2s}  {1:15.8f} {2:15.8f} {3:15.8f} \n'.format(symbol.short_name, *coord))

        with open(pj(calcFolder, self.fixedIndices_file), 'wt') as fp:
            fixedIndices = system['disassembler'].envIndices[
                system['environment'].getFixedIndices()] if 'environment' in system else []
            fp.write('LIST  ')
            for i in fixedIndices:
                fp.write('{} '.format(i + 1))

        if system['externalPressure']:
            with open(pj(calcFolder, self.pressure_file), 'a') as myfile:
                myfile.write(f"EXTERNAL_PRESSURE [GPa] {system['externalPressure']:10f}\n")

        atomTypes = structure.getAtomTypes()
        species = list(set(el.short_name for el in atomTypes))
        for i in species:
            atomTypes_file = i + self.atomIndices_file
            with open(pj(calcFolder, atomTypes_file), 'wt') as fp:
                fp.write('ATOMS_LIST  ')
                for j in range(len(atomTypes)):
                    if atomTypes[j].short_name == i:
                        fp.write('{} '.format(j + 1))

        return ''

    def isConverged(self, calcFolder: str):

        if not os.path.exists(pj(calcFolder, self.outputFile)):
            return False

        with open(pj(calcFolder, self.outputFile), 'rt') as f:
            content = f.read()
        if 'PROGRAM ENDED AT' not in content:
            logger.error('cp2k is not completely Done')
            return False
        return True

    def readOutput(self, system, calcFolder: str):

        new_structure = self.readStructure(system, calcFolder)
        EnergyHa = self.readEnergy(calcFolder)

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

    def readStructure(self, system, calcFolder: str):

        structure = system['structure']
        cell = structure.getCell()
        pbc = cell.getPBC()

        if os.path.exists(pj(calcFolder, self.out_cell_file)):
            with open(pj(calcFolder, self.out_cell_file), 'rt') as f:
                content_list = f.readlines()
                lattice = [float(x) for x in content_list[-1].split()[2:11]]
                lat = np.array([lattice[0:3], lattice[3:6], lattice[6:9]])
                cell = self.cellType(lat, pbc)
        else:
            with open(pj(calcFolder, self.outputFile), 'rt') as f:
               content = f.read()
            if ' CELL| Volume' in content:
                content_list = content.split('\n')
                for line in content_list:
                    if ' CELL| Vector a' in line:
                        lattice_a = [float(x) for x in line.split()[4:7]]
                    if ' CELL| Vector b' in line:
                        lattice_b = [float(x) for x in line.split()[4:7]]
                    if ' CELL| Vector c' in line:
                        lattice_c = [float(x) for x in line.split()[4:7]]
                lat = np.array([lattice_a, lattice_b, lattice_c])
                cell = self.cellType(lat, pbc)

        if os.path.exists(pj(calcFolder, self.out_geometry_file)):
            ase_struct = read(pj(calcFolder, self.out_geometry_file), index='-1')
            atomTypes = []
            for i in ase_struct.get_chemical_symbols():
                atomTypes.append(self.atomType(i))
            positions = ase_struct.get_positions()
            new_structure = self.structureType(atomTypes, positions, cell=cell)
        else:
            new_structure = self.structureType(structure.getAtomTypes(), structure.getCartesianCoordinates(), cell=cell)

        return new_structure

    def readEnergy(self, calcFolder: str):

        with open(pj(calcFolder, self.outputFile), 'rt') as f:
            content = f.read()
            content_list = content.split('\n')
        if ' ENERGY| Total FORCE_EVAL ( QS ) energy (a.u.):' in content:
            for line in content_list:
                if ' ENERGY| Total FORCE_EVAL ( QS ) energy (a.u.):' in line:
                    energy = float(line.split()[8])
        else:
            for line in content_list:
                if '  Total Energy               =' in line:
                    energy = float(line.split()[3])

        return energy