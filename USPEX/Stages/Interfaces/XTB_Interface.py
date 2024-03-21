"""
USPEX.Stages.XTB_Interface
===========================

"""

import logging
import numpy as np

from pathlib import Path
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

    inputFile, outputFile, errorFile = 'xtb.inp', 'output', 'error'
    geometry_file = 'uspex.gen'
    specific_file = 'xtb.inp_'

    out_geometry_file = 'xtbopt.gen'

    AtomicStructureRepresentation = None

    @classmethod
    def registerTypes(cls, AtomicStructureRepresentation):
        cls.AtomicStructureRepresentation = AtomicStructureRepresentation

    def __init__(self, tag: str, xtb_input: str = None, targetProperties: list = None, **kwargs):

        self.tag = tag
        if xtb_input is None:
            xtb_input = Path.cwd()/f'Specific/{self.specific_file}{tag}'

        assert xtb_input.exists(), f'Please, check path to xTB input. Now it is {xtb_input}'

        with open(xtb_input, 'r') as f:
            self.xtb_input = f.read()

        self.targetProperties = targetProperties

    def prepareLocalCalculation(self, system, calcFolder : Path):

        structure = system.getProperty('structure', extension='atomistic')
        cell = structure.getCell()
        with open(calcFolder/'pbc', 'wt') as f:
            f.write(' '.join(f'{c}' for c in cell.getPBC()))
        atoms = self.AtomicStructureRepresentation.toAtoms(structure)
        write_gen(calcFolder / self.geometry_file, atoms)

        content_to_write = ''
        disassembler = system.getProperty('disassembler', extension='atomistic')
        fixedIndices = disassembler.allFixedIndices
        if np.any(fixedIndices):
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

        with open(calcFolder/self.inputFile, 'wt') as dest:
            dest.write(total_content)

        return ''

    def isConverged(self, calcFolder: Path):

        if not calcFolder.joinpath(self.outputFile).exists():
            return False

        with open(calcFolder/self.outputFile, 'rt') as f:
            content = f.read()
        if 'finished run' not in content:
            logger.error('xtb is not completely Done')
            return False
        return True

    def readOutput(self, system, calcFolder: Path):

        factory = system.getFactory()
        result = factory()
        if 'structure' in self.targetProperties:
            with open(calcFolder / 'pbc', 'rt') as f:
                pbc = tuple(int(c) for c in f.read().split())
            atoms = read_gen(calcFolder / self.out_geometry_file)
            atoms.set_pbc(pbc)
            result.setProperty('structure', self.AtomicStructureRepresentation.fromAtoms(atoms), extension='atomistic')

        if 'energy' in self.targetProperties:
            result.setProperty('energy', self.readEnergyHa(calcFolder) * HARTREE_TO_EV)
        if 'enthalpy' in self.targetProperties:
            result.setProperty('enthalpy', self.readEnergyHa(calcFolder) * HARTREE_TO_EV)
        return result

    def readEnergyHa(self, calcFolder: Path):

        with open(calcFolder/self.outputFile, 'rt') as f:
            content = f.readlines()
            for i in content:
                if 'TOTAL ENERGY' in i:
                    EnergyHa = float(i.split()[3])

        return EnergyHa
