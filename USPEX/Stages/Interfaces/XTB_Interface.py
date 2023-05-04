"""
USPEX.Stages.XTB_Interface
===========================

"""

import logging
import os
from os.path import join as pj

from ase.io import read

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
    geometry_file = 'uspex.xyz'
    specific_file = 'xtb.inp_'

    out_geometry_file = 'xtbopt.xyz'

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

        with open(pj(calcFolder, self.inputFile), 'wt') as dest:
            dest.write(self.xtb_input)

        with open(pj(calcFolder, self.geometry_file), 'wt') as fp:
            fp.write('{}\n\n'.format(len(structure.getAtomTypes())))
            for i, (symbol, coord) in enumerate(zip(structure.getAtomTypes(), structure.getCartesianCoordinates())):
                fp.write('{0:2s}  {1:15.8f} {2:15.8f} {3:15.8f} \n'.format(symbol.short_name, *coord))

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

        new_structure, EnergyHa = self.readStructureAndEnergyHa(calcFolder, system.pop('pbc'))

        results = {}
        if 'structure' in self.targetProperties:
            results['structure'] = new_structure
        if 'energy' in self.targetProperties:
            results['energy'] = EnergyHa * HARTREE_TO_EV
        if 'enthalpy' in self.targetProperties:
            results['enthalpy'] = EnergyHa * HARTREE_TO_EV

        return results

    def readStructureAndEnergyHa(self, calcFolder: str, pbc):

        new_lattice = []
        ase_struct = read(pj(calcFolder, self.out_geometry_file), index='-1')
        atomTypes = []
        for i in ase_struct.get_chemical_symbols():
            atomTypes.append(self.atomType(i))
        positions = ase_struct.get_positions()
        cell = self.cellType.initFromCellVectors(pbc, new_lattice)
        new_structure = self.structureType(atomTypes, positions, cell=cell)

        with open(pj(calcFolder, self.out_geometry_file), 'rt') as f:
            content = f.readlines()
            EnergyHa = float(content[1].split()[1])

        return new_structure, EnergyHa