"""
USPEX.Stages.ORCA_Interface
===========================

"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)
HARTREE_TO_EV = 27.211386245988 #https://physics.nist.gov/cgi-bin/cuu/Value?hrev
GPA_TO_AU = 1.0/29421.015697
ANGSTROM_TO_BOHR = 1.0/0.529177210903

class ORCA_Interface:
    """
    Calculator for ORCA.
    Local running
    """
    DEFAULT_SLEEP_TIME = 30

    inputFile, outputFile, errorFile = 'orca.in', 'output', 'error'
    specific_file = 'orca.in_'

    out_geometry_file = 'orca.xyz'

    AtomicStructureRepresentation = None

    @classmethod
    def registerTypes(cls, AtomicStructureRepresentation):
        cls.AtomicStructureRepresentation = AtomicStructureRepresentation

    def __init__(self, tag: str,
                       orca_input: str = None,
                       targetProperties: list = None,
                       **kwargs):

        self.tag = tag
        orca_input = Path(orca_input) if orca_input is not None else Path.cwd()/f'Specific/{self.specific_file}{tag}'

        assert orca_input.exists(), f'Please, check path to ORCA input. Now it is {orca_input}'

        with open(orca_input, 'r') as f:
            self.orca_input = f.read()

        self.targetProperties = targetProperties

    def prepareLocalCalculation(self, system, calcFolder : Path):

        structure = system.getProperty('structure', extension='atomistic')
        cell = structure.getCell()
        with open(calcFolder/'pbc', 'wt') as f:
            f.write(' '.join(f'{c}' for c in cell.getPBC()))

        orca_input_lines = self.orca_input.split('\n')
        for i, line in enumerate(orca_input_lines):
            if 'xyz' in line:
                xyz_line_contents = line.split()
                xyz_line_index = i
        for i, j in enumerate(xyz_line_contents):
            if 'xyz' in j:
                charge = int(xyz_line_contents[i+1])
                multiplicity_request = int(xyz_line_contents[i+2])

        nelectrons = 0
        for symbol in structure.getAtomTypes():
            nelectrons += symbol.z
        total_nelectrons = nelectrons - charge

        if multiplicity_request <= 0:
            if (total_nelectrons % 2) == 0:
                multiplicity = 1
            else:
                multiplicity = 2
        else:
            if (multiplicity_request % 2) == 0:
                if (total_nelectrons % 2) == 0:
                    multiplicity = multiplicity_request - 1
                else:
                    multiplicity = multiplicity_request
            else:
                if (total_nelectrons % 2) == 0:
                    multiplicity = multiplicity_request
                else:
                    multiplicity = multiplicity_request + 1

        orca_input_lines[xyz_line_index] = '* xyz  {}  {}'.format(charge, multiplicity)

        disassembler = system.getProperty('disassembler', extension='atomistic')
        fixedIndices = disassembler.allFixedIndices
        for i, (symbol, coord) in enumerate(zip(structure.getAtomTypes(), structure.getCartesianCoordinates())):
            if i in fixedIndices:
                orca_input_lines.append('{0:2s}  {1:15.8f}$ {2:15.8f}$ {3:15.8f}$'
                                        .format(symbol.short_name, *coord))
            else:
                orca_input_lines.append('{0:2s}  {1:15.8f} {2:15.8f} {3:15.8f}'
                                        .format(symbol.short_name, *coord))

        orca_input_lines.append('*')

        final_input = [x for x in orca_input_lines if x != '']

        with open(calcFolder / self.inputFile, 'a') as fp:
            for i in final_input:
                fp.write(i+'\n')

        return ''

    def isConverged(self, calcFolder: Path) -> bool:
        if not calcFolder.joinpath(self.outputFile).exists():
            return False

        with open(calcFolder/self.outputFile, 'rt') as f:
            content = f.read()
        if ('TOTAL RUN TIME:' not in content) and \
                ('The optimization did not converge but reached the maximum' not in content):
            logger.error('ORCA is not completely Done')
            return False
        return True

    def readOutput(self, system, calcFolder: Path):
        new_structure = self.readStructure(system, calcFolder)
        EnergyHa = self.readEnergyHa(calcFolder)
        factory = system.getFactory()
        result = factory()

        if 'structure' in self.targetProperties:
            result.setProperty('structure', new_structure, extension='atomistic')
        if 'energy' in self.targetProperties:
            result.setProperty('energy', EnergyHa * HARTREE_TO_EV)
        if 'enthalpy' in self.targetProperties:
            result.setProperty('enthalpy', EnergyHa * HARTREE_TO_EV)
        return result

    def readStructure(self, system, calcFolder: Path):
        atomistic = system.getFactory().extensions['atomistic'].utility
        structure = system.getProperty('structure', extension='atomistic')

        if calcFolder.joinpath(self.out_geometry_file).exists():
            new_structure = self.AtomicStructureRepresentation.readXYZ(calcFolder/self.out_geometry_file)
        else:
            new_structure = atomistic.structureType(structure.getAtomTypes(), structure.getCartesianCoordinates(), cell=structure.getCell())

        return new_structure

    def readEnergyHa(self, calcFolder: Path):

        with open(calcFolder/self.outputFile, 'rt') as f:
            content = f.readlines()
            for i in content:
                if 'FINAL SINGLE POINT ENERGY' in i:
                    EnergyHa = float(i.split()[4])

        return EnergyHa