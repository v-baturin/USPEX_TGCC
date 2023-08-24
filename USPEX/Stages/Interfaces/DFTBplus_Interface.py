"""
USPEX.Stages.DFTBplus_Interface
===========================

"""

import logging
import numpy as np

from pathlib import Path
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

    inputFile, outputFile, errorFile = 'dftb_in.hsd', 'output', 'error'
    geometry_file = 'uspex.gen'
    specific_file = 'dftb_in.hsd_'
    kpoints_file = 'kpoints.uspex'
    pressure_file = 'pressure.uspex'
    movedAtoms_file = 'movedatoms.uspex'

    out_geometry_file = 'geo_end.gen'

    AtomicStructureRepresentation = None

    @classmethod
    def registerTypes(cls, AtomicStructureRepresentation):
        cls.AtomicStructureRepresentation = AtomicStructureRepresentation

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

        structure = system.getProperty('structure', extension='atomistic', suffix='intermediate')
        cell = structure.getCell()
        with open(calcFolder/'pbc', 'wt') as f:
            f.write(' '.join(f'{c}' for c in cell.getPBC()))

        atoms = self.AtomicStructureRepresentation.toAtoms(structure)
        write_gen(calcFolder / self.geometry_file, atoms)

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

        externalPressure = system.getProperty('externalPressure', suffix='origin')
        with open(calcFolder/self.pressure_file, 'wt') as f:
            if externalPressure:
                f.write(f"Pressure [Pa] = {externalPressure*10.0**9:10f}\n")
            else:
                f.write("")

        disassembler = system.getProperty('disassembler', extension='atomistic', suffix='intermediate')
        fixedIndices = disassembler.allFixedIndices
        if np.any(fixedIndices):
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
        with open(calcFolder / 'pbc', 'rt') as f:
            pbc = tuple(int(c) for c in f.read().split())
        atoms = read_gen(calcFolder / self.geometry_file)
        atoms.set_pbc(pbc)
        new_structure = self.AtomicStructureRepresentation.fromAtoms(atoms)

        EnergyHa = self.readEnergyHa(calcFolder)

        if 'structure' in self.targetProperties:
            system.setProperty('structure', new_structure, extension='atomistic', suffix=self.tag)
        if 'energy' in self.targetProperties:
            system.setProperty('energy', EnergyHa * HARTREE_TO_EV, suffix=self.tag)
        if 'enthalpy' in self.targetProperties:
            if new_structure.getCell().dim == 3:
                V = new_structure.getCell().getVolume()
                P = system.getProperty('externalPressure', suffix='origin')
                enthalpy = (EnergyHa + P*V*(ANGSTROM_TO_BOHR**3.0)*GPA_TO_AU) * HARTREE_TO_EV
                system.setProperty('enthalpy', enthalpy, suffix=self.tag)
            else:
                system.setProperty('enthalpy', EnergyHa * HARTREE_TO_EV, suffix=self.tag)

    def readEnergyHa(self, calcFolder: Path) -> float:
        with open(calcFolder/self.outputFile, 'rt') as f:
            content = f.readlines()
            for i in content:
                if 'Total Energy:' in i:
                    EnergyHa = float(i.split()[2])
        return EnergyHa
