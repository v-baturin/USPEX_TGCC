import logging
logger = logging.getLogger(__name__)

'''
@file        VCNEB_Calculator.py
@author:     Artem Samtsevich
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    samtsevichartem@gmail.com
@date        29 July 2016
@brief       Class for calculator of VASP
'''


__author__ = 'asamtsevich'

import numpy as np
import os
import shutil

from ase.io.vasp import read_vasp_out, write_vasp
from ase.atoms import Atoms
from ase.constraints import FixAtoms
from os.path import join as pj
from typing import List

from .Common.KPoints import KPoints, BadKPoints
from .Common.SHELL_Interface import SHELL_Interface

EV_PER_CUBIC_ANGSTREM_PER_GPA = 1/160.21766208


def split_up_data(data:List[str], out_size:int):
    '''
    Sometimes data in the OUTCAR is gleaned in the follows way:

      in kB    9069.66525 22624.04586166874.12221 14247.12053-60559.42376-37706.23723

    So, these values must be properly splitted and parsed
    '''
    if len(data) == out_size:  # proper output format
        res = np.array(data, dtype=float)
    else:  # problems in the output
        precision = len(data[-1].split('.')[-1])
        res = []
        for x in data:
            idx = [i for i, y in enumerate(x) if y == '.']
            k = 0
            for i in idx:
                res.append(float(x[k:i + precision + 1]))
                k = i + precision + 1
    assert len(res) == out_size
    return res


class VASP_Interface(SHELL_Interface):
    '''
    Calculator for VASP.
    Local running
    '''

    # working output files
    outcar_file = 'OUTCAR'
    oszicar_file = 'OSZICAR'
    contcar_file = 'CONTCAR'
    xml_file = 'vasprun.xml'

    # working output files
    incar_file = 'INCAR'
    kpoints_file = 'KPOINTS'
    poscar_file = 'POSCAR'
    potcar_file = 'POTCAR'


    _DEFAULT_SLEEP_TIME = 30
    structureType = None
    atomType = None
    cellType = None
    atomicDisassemblerType = None

    def __init__(self, tag: str, kresol: float, incar: str = None, potcarsPath: str = None, perturbate: bool = True,
                 vacuumSize = 10, targetObject: str = 'default', **kwargs):
        '''
        :param params: dictionary with parameters:
                * commandExecutable: str of executable command
                * kresol: float of K-points resolution
                * remote: dict of remote server params
                * taskManager: dict of task managers params
        :param step: int of current step
        '''

        super().__init__(**kwargs)

        if incar is not None:
            self.incar = incar
        else:
            self.incar = pj(os.getcwd(), f'Specific/INCAR_{tag}')

        assert os.path.exists(self.incar)

        if potcarsPath is not None:
            self.potcarsPath = potcarsPath
        else:
            self.potcarsPath = pj(os.getcwd(), 'Specific')

        self.kPoints = KPoints(kresol)
        self.failedSystems = []

        self.vacuumSize = vacuumSize
        self.targetObject = targetObject
        self.perturbate = perturbate


    def readOutput(self, system, calcFolder : str):
        self.readStructure(system, calcFolder)

        if self.targetObject == 'default':
            try:
                with open(pj(calcFolder, self.outcar_file)) as fp:
                    system['stressTensor'] = self.readPressureTensor(fp)
            except:
                logger.debug('Pressure tensor can\'t be find in output')
        elif self.targetObject == 'environment':
            pass

    def prepareLocalCalculation(self, system, calcFolder: str):
        '''
        :param system: our system
        :return:
        '''
        structure, disassembler = self.structureType.assemble(**system)
        system['disassembler'] = disassembler
        atomTypes = structure.getAtomTypes()
        system['symbolsOrder'] = np.argsort([el.short_name for el in atomTypes])

        coordinates = structure.getCartesianCoordinates()
        cell = structure.getRectifiedCell().getEnvelopeCell(coordinates, self.vacuumSize)
        coordinates = cell.center(coordinates)

        with open(pj(calcFolder, self.inputFile), 'wt') as f:
            pass

        ############################## INCAR ##################################
        shutil.copy2(self.incar, pj(calcFolder, self.incar_file))

        if system['externalPressure']:
            with open(pj(calcFolder, self.incar_file), 'a') as myfile:
                myfile.write(f"\nPSTRESS={10 * system['externalPressure']:10f}\n")
        if calcFolder in self.failedSystems:
            with open(pj(calcFolder, self.incar_file), 'a') as myfile:
                myfile.write('ISYM=0\n')

        ############################# POTCAR ##################################
        if os.path.exists(pj(calcFolder, 'POTCAR')):
            os.remove(pj(calcFolder, 'POTCAR'))

        for atomType in np.unique([el.short_name for el in atomTypes]):
            potcarPath = pj(self.potcarsPath, f'POTCAR_{atomType}')
            os.system(f'cat {potcarPath} >>  {calcFolder}/POTCAR ')

        ############################# POSCAR ##################################
        if self.perturbate:
            coordinates += 0.1 * (np.random.rand(len(structure), 3) - 0.5)

        with open(pj(calcFolder, self.poscar_file), 'wt') as f:
            if self.targetObject == 'default':
                atoms = Atoms([el.short_name for el in atomTypes], coordinates, cell = cell.getCellVectors())
                if 'environment' in system:
                    indices = disassembler.envIndices[system['environment'].getFixedIndices()]
                    atoms.set_constraint(FixAtoms(indices=indices))
                write_vasp(f, atoms, label=f"EA{system['ID']}", sort=True, direct=True, vasp5=True, long_format=False)
            elif self.targetObject == 'environment':
                environment = system['environment'].getStructure()
                atoms = Atoms([el.short_name for el in environment.getAtomTypes()], environment.getCartesianCoordinates(), cell = cell.getCellVectors())
                write_vasp(f, atoms, label=f"EA{system['ID']}", sort=True, direct=True, vasp5=True, long_format=False)

        ############################# KPOINTS #################################
        try:
            kPoints = self.kPoints.build(cell)
        except BadKPoints:
            # This LATTICE is extremely wrong, let's skip it from now
            logger.info('K-points cannot be built, so it\'s set as   [1, 1, 1]')
            kPoints = [1, 1, 1]

        with open(pj(calcFolder, self.kpoints_file), 'w') as fp:
            fp.write('EA\n0\nGamma\n')
            fp.write('%4d %4d %4d\n' % tuple(kPoints))

        # copyfile('POSCAR', 'POSCAR-G' + str(system.generation) + '-N' + str(system.index) + '-S' + str(system.step))

        ############################# LDAU #################################
        ####################################################################
        # TODO LDAU implementation
        # if isfield( ORG_STRUC, 'ldaU') & sum(norm(ORG_STRUC.ldaU)) > 0
        #     fp = fopen('INCAR_LDAUPart', 'w');
        #     fprintf(fp,'\n\n');
        #
        #
        #     ldaUPrint=zeros(3,size(ORG_STRUC.ldaU,2));
        #     ldaUPrint(1,:)=ORG_STRUC.ldaU(1,:);
        #     ldaUPrint(2,:)=ORG_STRUC.ldaU(2,:);
        #
        #     for i =1:length(POP_STRUC.POPULATION(Ind_No).numIons)
        #         if  ORG_STRUC.ldaU(i)>0
        #             ldaUPrint(3,i)= 2;
        #         else
        #             ldaUPrint(3,i)=-1;
        #         end
        #     end
        #     isPrint=zeros(1,length(ORG_STRUC.ldaU));
        #     for i = 1:length(POP_STRUC.POPULATION(Ind_No).numIons)
        #         isPrint(i)=ORG_STRUC.ldaU(1,i)*POP_STRUC.POPULATION(Ind_No).numIons(i);
        #     end
        #     if sum(isPrint)>0
        #         fprintf(fp,'LDAU=.True.\n');
        #         fprintf(fp,'LDAUTYPE=2 \n');
        #         fprintf(fp,'LDAUPRINT =2\n');
        #         fprintf(fp,'LDAUL= ');
        #         for i = 1:length(POP_STRUC.POPULATION(Ind_No).numIons)
        #             if  POP_STRUC.POPULATION(Ind_No).numIons(i) > 0
        #                 fprintf(fp,'%2d ',ldaUPrint(3,i));
        #             end
        #         end
        #         fprintf(fp,'\nLDAUU= ');
        #         for i = 1:length(POP_STRUC.POPULATION(Ind_No).numIons)
        #             if  POP_STRUC.POPULATION(Ind_No).numIons(i) > 0
        #                 fprintf(fp,'%2d ',ldaUPrint(1,i));
        #             end
        #         end
        #         fprintf(fp,'\nLDAUJ= ');
        #         for i = 1:length(POP_STRUC.POPULATION(Ind_No).numIons)
        #             if  POP_STRUC.POPULATION(Ind_No).numIons(i) > 0
        #                 fprintf(fp,'%2d ',ldaUPrint(2,i));
        #             end
        #         end
        #         fprintf(fp,'\n');
        #
        #     end
        #
        #     fclose(fp);
        #     [nothing, nothing] = unix('cat INCAR_LDAUPart >> INCAR');
        # end


############reading part

    def isConverged(self, calcFolder : str):
        '''
        :param SYSTEM:
        :return: (bool) whether system calculation converged
        '''

        if not (os.path.exists(pj(calcFolder, self.outcar_file)) and
                os.path.exists(pj(calcFolder, self.oszicar_file)) and
                os.path.exists(pj(calcFolder, self.contcar_file))):
            return False

        # Checking whether converge
        NELM = -1
        row_number = None
        with open(pj(calcFolder, self.oszicar_file), 'r') as f:
            content = f.readlines()
            for i in range(len(content)):
                if content[i].find(' F= ') >= 0:
                    row_number = i
            if row_number is None:
                return False

            # Read previous line to check the number of SCF steps:
            vaspSCFsteps = int(content[row_number - 1].split(':')[1].split()[0].strip())

        with open(pj(calcFolder, self.outcar_file), 'r') as f:
            for line in f:
                if line.find(' NELM ') >= 0:
                    NELM = int(line.split(' = ')[1].split()[0].replace(';', '').strip())
                    break

        # The calculation is considered successful in case vaspSCFsteps < NELM:
        if vaspSCFsteps < NELM:
            return True
        else:
            logger.error('VASP SCF is not converged.')
            shutil.copy2(pj(calcFolder, self.outcar_file), f'{pj(calcFolder, "ERROR")}-{self.outcar_file}')
            self.failedSystems.append(calcFolder)
            return False

    def readPressureTensor(self, filename='OUTCAR', index=-1):
        '''

        :param filename:
        :param index:
        :return:
        '''

        if isinstance(filename, str):
            f = open(filename)
        else:  # Assume it's a file-like object
            f = filename
        content = f.readlines()

        target = []
        for line in content:
            if 'in kB' in line:
                stress = split_up_data(line.split()[2:], 6)
                thisTarget = np.diag(stress[0:3])
                thisTarget[0, 1] = thisTarget[1, 0] = stress[3]
                thisTarget[1, 2] = thisTarget[2, 1] = stress[4]
                thisTarget[0, 2] = thisTarget[2, 0] = stress[5]
                target.append(thisTarget)

        if isinstance(index, int):
            return target[index]
        else:
            step = index.step or 1
            if step > 0:
                start = index.start or 0
                if start < 0:
                    start += len(target)
                stop = index.stop or len(target)
                if stop < 0:
                    stop += len(target)
            else:
                if index.start is None:
                    start = len(target) - 1
                else:
                    start = index.start
                    if start < 0:
                        start += len(target)
                if index.stop is None:
                    stop = -1
                else:
                    stop = index.stop
                    if stop < 0:
                        stop += len(target)
            return [target[i] for i in range(start, stop, step)]

    def readDielectricConstant(self, filename='OUTCAR', index=-1):
        '''
        reads dielectric susceptibility tensor from OUTCAR file. Format:
        MACROSCOPIC STATIC DIELECTRIC TENSOR (including local field effects in DFT)
        -------------------------------------
                  4.061     0.000     0.000
                  0.000     4.061     0.000
                  0.000     0.000     4.061
        -------------------------------------
        MACROSCOPIC STATIC DIELECTRIC TENSOR IONIC CONTRIBUTION
        -------------------------------------
                 13.471     0.000     0.000
                  0.000    13.471     0.000
                  0.000     0.000    13.462
        ------------------------------------
        '''
        d_s = np.zeros((6, 1), dtype=float)
        d_s_ion = np.zeros((6, 1), dtype=float)
        with open(filename, 'rb') as f:
            content = f.readlines()

        offset = len(content)
        for n, line in enumerate(reversed(content)):
            if 'MACROSCOPIC STATIC DIELECTRIC TENSOR IONIC' in line:
                offset_ion = offset - n
                continue
            if 'MACROSCOPIC STATIC DIELECTRIC TENSOR (including' in line:
                offset -= n
                break

        # print content[offset - 1]
        # print content[offset_ion - 1]
        '''
        d_s[0] d_s[3] d_s[4]
        d_s[3] d_s[1] d_s[5]
        d_s[4] d_s[5] d_s[2]

        '''
        d_s[0], d_s[3], d_s[4] = content[offset + 1].split()
        d_s[3], d_s[1], d_s[5] = content[offset + 2].split()
        d_s[4], d_s[5], d_s[2] = content[offset + 3].split()

        d_s_ion[0], d_s_ion[3], d_s_ion[4] = content[offset_ion + 1].split()
        d_s_ion[3], d_s_ion[1], d_s_ion[5] = content[offset_ion + 2].split()
        d_s_ion[4], d_s_ion[5], d_s_ion[2] = content[offset_ion + 3].split()

        d_s += d_s_ion

        imaginary_f = []
        for line in content:
            if 'f/i' in line:
                imaginary_f.append(abs(float(line.split()[2])))
        if imaginary_f[-1] > 0.3 or imaginary_f[-2] > 0.3 or imaginary_f[-3] > 0.3:
            d_s = np.zeros((6, 1), dtype=float)

        return d_s

    def readStructure(self, system, calcFolder : str):
        cell = system['cell']
        disassembler = system['disassembler']
        del system['disassembler']
        symbolsOrder = system['symbolsOrder']
        del system['symbolsOrder']

        tmp = read_vasp_out(pj(calcFolder, self.outcar_file))
        if tmp:
            if self.targetObject == 'default':
                tmp_positions = tmp.get_positions()
                positions = np.empty(tmp_positions.shape, dtype=float)
                tmp_symbols = tmp.get_chemical_symbols()
                atomTypes = np.empty(len(tmp_symbols), dtype=self.atomType)
                for i, symbol, position in zip(symbolsOrder, tmp_symbols, tmp_positions):
                    positions[i] = position
                    atomTypes[i] = self.atomType(symbol)

                cell = self.cellType(tmp.get_cell().array, cell.getPBC()).getEnvelopeCell(positions, 0)
                positions = cell.center(positions)
                system.update(disassembler.disassemble(self.structureType(atomTypes, positions, cell=cell)))
                system['enthalpy'] = float(tmp.get_calculator().results['energy']) + \
                                     tmp.get_volume() * system['externalPressure'] * EV_PER_CUBIC_ANGSTREM_PER_GPA
                # system.forces = np.copy(tmp.get_calculator().results['forces'])
            elif self.targetObject == 'environment':
                system['environmentEnthalpy'] = float(tmp.get_calculator().results['energy']) + \
                                  tmp.get_volume() * system['externalPressure'] * EV_PER_CUBIC_ANGSTREM_PER_GPA
                


    # TODO check this out
    def readEnergy(self):
        energy_free, energy_zero = 0, 0
        if all:
            energy_free = []
            energy_zero = []
        for line in open(self.outcar_file, 'r'):
            # Free energy
            if line.lower().startswith('  free  energy   toten'):
                if all:
                    energy_free.append(float(line.split()[-2]))
                else:
                    energy_free = float(line.split()[-2])

            # Extrapolated zero point energy
            if line.startswith('  energy  without entropy'):
                if all:
                    energy_zero.append(float(line.split()[-1]))
                else:
                    energy_zero = float(line.split()[-1])
        return [energy_free, energy_zero]

    def readForces(self, atoms, all=False):
        """Method that reads forces from OUTCAR file.

        If 'all' is switched on, the forces for all ionic steps
        in the OUTCAR file be returned, in other case only the
        forces for the last ionic configuration is returned."""

        file = open(self.outcar_file, 'r')
        lines = file.readlines()
        file.close()
        n = 0
        if all:
            all_forces = []
        for line in lines:
            if line.rfind('TOTAL-FORCE') > -1:
                forces = []
                for i in range(len(atoms)):
                    forces.append(np.array([float(force) for force in
                                            lines[n + 2 + i].split()[3:6]]))
                if all:
                    all_forces.append(np.array(forces)[self.resort])
            n += 1
        if all:
            return np.array(all_forces)
        else:
            return np.array(forces)[self.resort]

    def readDipoleMoment(self):
        dipoleMoment = np.zeros([1, 3])
        for line in open(self.outcar_file, 'r'):
            if line.rfind('dipolmoment') > -1:
                dipoleMoment = np.array([float(f) for f in line.split()[1:4]])
        return dipoleMoment

    def readFermi(self):
        energyFermi = None
        for line in open(self.outcar_file, 'r'):
            if line.rfind('E-fermi') > -1:
                energyFermi = float(line.split()[2])
        return energyFermi

    @classmethod
    def registerTypes(cls, structureType, atomType, cellType, atomicDisassemblerType):
        cls.structureType = structureType
        cls.atomType = atomType
        cls.cellType = cellType
        cls.atomicDisassemblerType = atomicDisassemblerType
