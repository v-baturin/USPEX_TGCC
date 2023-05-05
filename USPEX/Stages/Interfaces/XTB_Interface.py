"""
USPEX.Stages.XTB_Interface
===========================

"""

import logging
import os
from os.path import join as pj
import numpy as np

from ase import Atoms
from ase.io.gen import read_gen, write_gen

logger = logging.getLogger(__name__)
HARTREE_TO_EV = 27.211386245988 #https://physics.nist.gov/cgi-bin/cuu/Value?hrev
GPA_TO_AU = 1.0/29421.015697
ANGSTROM_TO_BOHR = 1.0/0.529177210903

class XTB_Interface:
    """
    Calculator for xTB.
    Local running
    """
    DEFAULT_SLEEP_TIME = 30
    structureType = None
    atomType = None
    cellType = None

    inputFile, outputFile, errorFile = 'xtb.inp', 'output', 'error'
    geometry_file = 'uspex.gen'
    specific_file = 'xtb.inp_'

    out_geometry_file = 'xtbopt.gen'

    @classmethod
    def registerTypes(cls, structureType, atomType, cellType):
        cls.structureType = structureType
        cls.atomType = atomType
        cls.cellType = cellType

    def __init__(self, tag: str, xtb_input: str = None, targetProperties: list = None, **kwargs):

        self.tag = tag
        if xtb_input is None:
            xtb_input = pj(os.getcwd(), f'Specific/{self.specific_file}{tag}')

        assert os.path.exists(xtb_input)

        with open(xtb_input, 'r') as f:
            self.xtb_input = f.read()

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

        content_to_write = ''
        fixedIndices = system['disassembler'].envIndices[
            system['environment'].getFixedIndices()] if 'environment' in system else None
        if fixedIndices is not None:
            content_to_write += '$fix\n'
            content_to_write += 'atoms: '
            onebased_fixedIndices = fixedIndices + 1
            indices = np.where(np.diff(onebased_fixedIndices) != 1)[0] + 1
            groups = np.split(onebased_fixedIndices, indices)
            for i, group in enumerate(groups):
                if len(group) == 1:
                    content_to_write += str(group[0])
                else:
                    content_to_write += f"{group[0]}-{group[-1]}"
                if i < len(groups) - 1:
                    content_to_write += ','
            content_to_write += '\n$end\n'
        total_content = self.xtb_input + '\n' + content_to_write + '\n'

        with open(pj(calcFolder, self.inputFile), 'wt') as dest:
            dest.write(total_content)

        return ''

    def isConverged(self, calcFolder: str):

        if not os.path.exists(pj(calcFolder, self.outputFile)):
            return False

        with open(pj(calcFolder, self.outputFile), 'rt') as f:
            content = f.read()
        if 'finished run' not in content:
            logger.error('xtb is not completely Done')
            return False
        return True

    def readOutput(self, system, calcFolder: str):

        results = {}
        if 'structure' in self.targetProperties:
            results['structure'] = self.readStructure(calcFolder, system.pop('pbc'))
        if 'energy' in self.targetProperties:
            results['energy'] = self.readEnergyHa(calcFolder) * HARTREE_TO_EV
        if 'enthalpy' in self.targetProperties:
            results['enthalpy'] = self.readEnergyHa(calcFolder) * HARTREE_TO_EV

        return results

    def readStructure(self, calcFolder: str, pbc):

        LOCAL_VACUUM = 4.0

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
#        if sum(pbc) == 0:
#            xsize = np.amax(ase_struct.get_positions()[:, 0]) - np.amin(ase_struct.get_positions()[:, 0])
#            ysize = np.amax(ase_struct.get_positions()[:, 1]) - np.amin(ase_struct.get_positions()[:, 1])
#            zsize = np.amax(ase_struct.get_positions()[:, 2]) - np.amin(ase_struct.get_positions()[:, 2])
#            new_lattice = np.array([[xsize + LOCAL_VACUUM, 0.0, 0.0], [0.0, ysize + LOCAL_VACUUM, 0.0],\
#                                    [0.0, 0.0, zsize + LOCAL_VACUUM]])
#        else:
#            new_lattice = ase_struct.cell[:].copy()
#        cell = self.cellType(new_lattice, pbc)
        new_structure = self.structureType(atomTypes, positions, cell=cell)

        return new_structure

    def readEnergyHa(self, calcFolder: str):

        with open(pj(calcFolder, self.outputFile), 'rt') as f:
            content = f.readlines()
            for i in content:
                if 'TOTAL ENERGY' in i:
                    EnergyHa = float(i.split()[3])

        return EnergyHa