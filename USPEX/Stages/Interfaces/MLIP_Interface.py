"""
USPEX.Stages.MLIP_Interface
===========================

.. codeauthor:: Michele Galasso <m.galasso@yandex.com>

"""
import logging
import os
import shutil
import numpy as np
from os.path import join as pj
from ase.atoms import Atoms

from ...Presets import udateSystemWithPrefix as usp


logger = logging.getLogger(__name__)
EV_PER_CUBIC_ANGSTREM_PER_GPA = 1 / 160.21766208


class MLIP_Interface:
    '''
    Calculator for MLIP.
    Local running
    '''


    inputFile, outputFile, errorFile = 'input', 'output', 'error'

    # working input files
    in_cfg_file = 'for_relax.cfg'

    # working output files
    out_cfg_file = 'relaxed.cfg_0'
    out_sampled_file = 'sampled.cfg_0'

    DEFAULT_SLEEP_TIME = 10
    structureType = None
    atomType = None
    cellType = None
    atomicDisassemblerType = None

    @classmethod
    def registerTypes(cls, structureType, atomType, cellType, atomicDisassemblerType):
        cls.structureType = structureType
        cls.atomType = atomType
        cls.cellType = cellType
        cls.atomicDisassemblerType = atomicDisassemblerType

    def __init__(self, tag: str, input: str = None, potential: str = None, vacuumSize = 10,
                 environmentStyle=None, inStyle=None, targetProperties: list = None, **kwargs):

        self.tag = tag
        self.tmp = f'tmp_{tag}'
        if input is not None:
            self.input = input
        else:
            self.input = pj(os.getcwd(), f'Specific/input_{tag}.ini')

        if potential is not None:
            self.potential = potential
        else:
            self.potential = pj(os.getcwd(), 'Specific/potential.mtp')
        self.vacuumSize = vacuumSize
        self.targetProperties = targetProperties if targetProperties is not None else ['structure', 'enthalpy']
        self.environmentStyle = environmentStyle
        self.inStyle = inStyle

    def prepareLocalCalculation(self, system, calcFolder: str):
        structure, disassembler = self.atomicDisassemblerType.assemble(**system,
                                                                       style=self.environmentStyle,
                                                                       inStyle=self.inStyle,
                                                                       vacuumSize=self.vacuumSize)
        system[self.tmp]['disassembler'] = disassembler
        cell = structure.getCell()
        system['pbc'] = cell.getPBC()

        # create empty input file
        with open(pj(calcFolder, self.inputFile), 'wt') as f:
            pass

        # cfg file
        self.savecfg(pj(calcFolder, self.in_cfg_file), structure, system)

        # input file
        shutil.copy2(self.input, calcFolder)

        # potential file
        shutil.copy2(self.potential, calcFolder)

    def isConverged(self, calcFolder: str):
        if os.path.isfile(pj(calcFolder, self.out_cfg_file)):
            with open(pj(calcFolder, self.out_cfg_file), 'r') as f:
                content = f.read()
            if content:
                return True
            # if the structure ended up unrelaxed because of extrapolation
            elif os.path.isfile(pj(calcFolder, self.out_sampled_file)):
                with open(pj(calcFolder, self.errorFile)) as stderr:
                    content = stderr.read()
                if not content:
                    return True
        return False

    def readOutput(self, system, calcFolder: str):
        data = self.readcfg(pj(calcFolder, self.out_cfg_file))
        if 'structure' in self.targetProperties:
            usp(system, system[self.tmp].pop('disassembler').disassemble(data['structure']), 'system', self.environmentStyle)
        if 'enthalpy' in self.targetProperties:
            enthalpy = data['energy'] + \
                       data['structure'].getCell().getVolume() * system['externalPressure'] * EV_PER_CUBIC_ANGSTREM_PER_GPA
            usp(system, enthalpy, 'enthalpy', self.environmentStyle)
        if 'stressTensor' in self.targetProperties:
            stress_tensor = np.zeros((3, 3))
            stress_tensor[0, 0] = data['stresses'][0]
            stress_tensor[1, 1] = data['stresses'][1]
            stress_tensor[2, 2] = data['stresses'][2]
            stress_tensor[1, 2] = data['stresses'][3]
            stress_tensor[2, 1] = data['stresses'][3]
            stress_tensor[0, 2] = data['stresses'][4]
            stress_tensor[2, 0] = data['stresses'][4]
            stress_tensor[0, 1] = data['stresses'][5]
            stress_tensor[1, 0] = data['stresses'][5]
            system['stressTensor'] = stress_tensor
            usp(system, stress_tensor, 'stressTensor', self.environmentStyle)

        # else:
        #     ID = system['ID']
        #     logger.info(f'structure {ID} led to extrapolation and will be discarded.')
        #     # system['structure'].set_cell(np.identity(3) * system['structure'].minVectorLength * 0.9)
        #     system['enthalpy'] = 1000


    def readcfg(self, filename):
        with open(filename, 'r') as f:
            lat = np.zeros((3, 3))
            types = None
            pos = None
            energy = None
            forces = None
            stresses = None
            size = -1
            mode = -1
            line = f.readline()
            while line:
                line = line.upper()
                line = line.strip()
                if mode == 0:
                    if line.startswith('SIZE'):
                        line = f.readline()
                        size = int(line.strip())
                        types = np.zeros(size, dtype=int)
                        pos = np.zeros((size, 3))
                    elif line.startswith('SUPERCELL'):
                        line = f.readline()
                        vals = line.strip().split()
                        lat[0, :] = vals[0:3]
                        line = f.readline()
                        vals = line.strip().split()
                        lat[1, :] = vals[0:3]
                        line = f.readline()
                        vals = line.strip().split()
                        lat[2, :] = vals[0:3]
                    elif line.startswith('ATOMDATA'):
                        if line.endswith('FZ'):
                            forces = np.zeros((size, 3))
                        for i in range(size):
                            line = f.readline()
                            vals = line.strip().split()
                            types[i] = int(vals[1])
                            pos[i, :] = vals[2:5]
                            if forces is not None:
                                forces[i, :] = vals[5:8]
                    elif line.startswith('ENERGY'):
                        line = f.readline()
                        energy = float(line.strip())
                    elif line.startswith('PLUSSTRESS'):
                        line = f.readline()
                        vals = line.strip().split()
                        stresses = np.zeros(6)
                        stresses[:] = vals[0:6]
                if line.startswith('BEGIN_CFG'):
                    mode = 0
                elif line.startswith('END_CFG'):
                    break
                line = f.readline()

        cell = self.cellType(lat, (1, 1, 1))
        return dict(
            structure=self.structureType([self.atomType(int(n)) for n in types], pos, cell=cell),
            energy=energy,
            forces=forces,
            stresses=stresses
        )

    @staticmethod
    def savecfg(filename, structure, system):
        with open(filename, 'w') as f:
            atstr1 = 'AtomData:  id type      cartes_x      cartes_y      cartes_z           fx          fy          fz\n'
            atstr2 = 'AtomData:  id type      cartes_x      cartes_y      cartes_z\n'
            size = len(structure)
            f.write('BEGIN_CFG\n')
            f.write('Size\n')
            f.write(f'   {size}\n')
            f.write('SuperCell\n')
            for i in range(3):
                lat = structure.getCell().getCellVectors()
                f.write(' %13f %13f %13f\n' % (lat[i, 0], lat[i, 1], lat[i, 2]))
            if 'forces' in system:
                f.write(atstr1)
            else:
                f.write(atstr2)
            atomTypes = structure.getAtomTypes()
            positions = structure.getCartesianCoordinates()
            for i in range(size):
                if 'forces' in system:
                    f.write('         %4d %4d %13f %13f %13f %11.8e %11.8e %11.8e\n' %
                            (i + 1, atomTypes[i].z, positions[i, 0], positions[i, 1], positions[i, 2],
                             system['forces'][i, 0], system['forces'][i, 1], system['forces'][i, 2]))
                else:
                    f.write('         %4d %4d %13f %13f %13f\n' %
                            (i + 1, atomTypes[i].z, positions[i, 0], positions[i, 1], positions[i, 2]))
            if 'energy' in system:
                f.write(' Energy\n   %20f\n' % system['energy'])
            if 'stresses' in system:
                f.write(' PlusStress:  xx           yy           zz           yz           xz           xy\n')
                f.write('         %11f %11f %11f %11f %11f %11f\n' %
                        (system['stresses'][0], system['stresses'][1], system['stresses'][2],
                         system['stresses'][3], system['stresses'][4], system['stresses'][5]))
            f.write('END_CFG\n')
