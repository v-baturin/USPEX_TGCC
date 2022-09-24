"""
USPEX.Stages.VASP_Interface
===========================

"""

import logging
import numpy as np
import os
import shutil
from ase.io.vasp import read_vasp_out, read_vasp_xml, write_vasp
from ase.io import ParseError as aseParseError
from ase.atoms import Atoms
from ase.constraints import FixAtoms
from os.path import join as pj
from typing import List

from .KPoints import KPoints, BadKPoints

logger = logging.getLogger(__name__)
EV_PER_CUBIC_ANGSTREM_PER_GPA = 1/160.21766208
STRUCTURE_STYLES = ['structure', 'adjustedStructure']
ENTHALPY_STYLES = ['enthalpy', 'adjustedEnthalpy', 'environmentEnthalpy', 'lowerEnvironmentEnthalpy', 'upperEnvironmentEnthalpy']
ENERGY_STYLES = ['energy', 'adjustedEnergy', 'environmentEnergy', 'lowerEnvironmentEnergy', 'upperEnvironmentEnergy']
STRESS_TENSOR_STYLES = ['stressTensor', 'environmentStressTensor', 'lowerEnvironmentStressTensor', 'upperEnvironmentStressTensor']


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


class VASP_Interface:
    '''
    Calculator for VASP.
    Local running
    '''

    inputFile, outputFile, errorFile = 'input', 'output', 'error'

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


    DEFAULT_SLEEP_TIME = 30
    structureType = None
    atomType = None
    cellType = None
    atomicDisassemblerType = None

    def __init__(self, tag: str, kresol: float, incar: str = None, potcarsPath: str = None, perturbate: bool = True,
                 vacuumSize = 10, targetProperties: list = None, adjustEnvironment: bool = False, **kwargs):
        '''
        :param params: dictionary with parameters:
                * commandExecutable: str of executable command
                * kresol: float of K-points resolution
                * remote: dict of remote server params
                * taskManager: dict of task managers params
        :param step: int of current step
        '''

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
        self.targetProperties = targetProperties if targetProperties is not None else ['structure', 'enthalpy']
        self.perturbate = perturbate
        self.adjustEnvironment = adjustEnvironment


    def prepareLocalCalculation(self, system, calcFolder: str):
        '''
        :param system: our system
        :return:
        '''
        molecules, cell = system['molecules'], system['cell']
        environment = system.get('environment')
        if self.adjustEnvironment:
            logger.debug('"adjustEnvironment" option was enabled , building the adjusted system')
            if 'adjustedMolecules' not in system or 'adjustedCell' not in system or 'adjustedEnvironment' not in system:
                molecules, cell, environment = system['molecules'], system['cell'], system['environment']
                logger.debug(f'cellVectors: {cell.getCellVectors()} (film), {environment.getStructure().getCell().getCellVectors()} (substrate)')
                adjustedMolecules, adjustedCell, adjustedEnvironment = type(environment).adjustSystem(molecules, cell, environment)
                system['adjustedMolecules'], system['adjustedCell'], system['adjustedEnvironment'] = adjustedMolecules, adjustedCell, adjustedEnvironment
                system['adjustedSupercellFactor'] = int(len(adjustedMolecules) / len(molecules))
            else:
                logger.debug('Adjusted data was found in system, proceeding with it')
                adjustedMolecules, adjustedCell, adjustedEnvironment = system['adjustedMolecules'], system['adjustedCell'], system['adjustedEnvironment']
            molecules, cell, environment = adjustedMolecules, adjustedCell, adjustedEnvironment
        if 'noEnvironment' in self.targetProperties:
            logger.debug('"noEnvironment" option was found in targetProperties, proceeding without environment')
            structure, disassembler = self.structureType.assemble(molecules, cell, vacuumSize=self.vacuumSize)
        else:
            logger.debug('Assembling the structure')
            structure, disassembler = self.structureType.assemble(molecules, cell, environment, vacuumSize=self.vacuumSize)


        system['disassembler'] = disassembler
        atomTypes = structure.getAtomTypes()
        system['symbolsOrder'] = np.argsort([el.short_name for el in atomTypes])
        cell = structure.getCell()
        system['assembledCell'] = cell
        coordinates = structure.getCartesianCoordinates()

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


        ############################# POSCAR ##################################
        if self.perturbate:
            coordinates += 0.1 * (np.random.rand(len(structure), 3) - 0.5)

        with open(pj(calcFolder, self.poscar_file), 'wt') as f:
            for onlyEnvironment, getStructure in environment.processingStyles.items():
                if onlyEnvironment in self.targetProperties:
                    envStructure = getattr(environment, getStructure)()
                    coordinates = envStructure.getCartesianCoordinates()
                    cell = envStructure.getRectifiedCell().getEnvelopeCell(coordinates, self.vacuumSize)
                    coordinates = cell.center(coordinates)
                    structure = type(envStructure)(envStructure.getAtomTypes(), coordinates, cell)
                    fixedIndices = disassembler.envIndices[environment.getFixedIndices()]
                else:
                    fixedIndices = []

            atoms = Atoms([el.short_name for el in structure.getAtomTypes()], structure.getCartesianCoordinates(), cell = structure.getCell().getCellVectors())
            if fixedIndices:
                atoms.set_constraint(FixAtoms(indices=fixedIndices))
            write_vasp(f, atoms, label=f"EA{system['ID']}", sort=True, direct=True, vasp5=True, long_format=False)
            
        ############################# POTCAR ##################################
        if os.path.exists(pj(calcFolder, 'POTCAR')):
            os.remove(pj(calcFolder, 'POTCAR'))

        for atomType in np.unique(atoms.symbols):
            potcarPath = pj(self.potcarsPath, f'POTCAR_{atomType}')
            os.system(f'cat {potcarPath} >>  {calcFolder}/POTCAR ')

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

    def readOutput(self, system, calcFolder : str):
        try:
            aseStructure = read_vasp_out(pj(calcFolder, self.outcar_file))
        except (KeyError, aseParseError):
            aseStructure = list(read_vasp_xml(pj(calcFolder, self.xml_file)))[-1]
        if aseStructure:
            if any([item in self.targetProperties for item in STRUCTURE_STYLES]):
                self.readStructure(system, aseStructure)
        for enthalpyStyle in ENERGY_STYLES:
            if enthalpyStyle in self.targetProperties:
                system[enthalpyStyle] = float(aseStructure.get_calculator().results['energy']) + \
                           aseStructure.get_volume() * system['externalPressure'] * EV_PER_CUBIC_ANGSTREM_PER_GPA
        for energyStyle in ENERGY_STYLES:
            if energyStyle in self.targetProperties:
                system[energyStyle] = float(aseStructure.get_calculator().results['energy'])
        if 'forces' in self.targetProperties:
                system['forces'] = np.copy(aseStructure.get_calculator().results['forces'])
        with open(pj(calcFolder, self.outcar_file), 'rt') as fp:
            content = fp.readlines()
        for stressTensorStyle in STRESS_TENSOR_STYLES:
            if stressTensorStyle in self.targetProperties:
                system[stressTensorStyle] = self.readPressureTensor(content)
        if 'dielectricTensor' in self.targetProperties:
            system['dielectricTensor'] = self.readDielectricProperties(content)
        if 'dipoleMoment' in self.targetProperties:
            system['dipoleMoment'] = self.readDipoleMoment(content)
        if 'energyFermi' in self.targetProperties:
            system['energyFermi'] = self.readFermi(content)
        if 'elasticConstants' in self.targetProperties:
            system['elasticMatrix'] = self.readElasticMatrix(content)

    def readStructure(self, system, aseStructure):
        assembledCell = system.pop('assembledCell')
        disassembler = system.pop('disassembler')
        symbolsOrder = system.pop('symbolsOrder')
        tmp_positions = aseStructure.get_positions()
        positions = np.empty(tmp_positions.shape, dtype=float)
        tmp_symbols = aseStructure.get_chemical_symbols()
        atomTypes = np.empty(len(tmp_symbols), dtype=self.atomType)
        for i, symbol, position in zip(symbolsOrder, tmp_symbols, tmp_positions):
            positions[i] = position
            atomTypes[i] = self.atomType(symbol)
        cell = self.cellType(aseStructure.get_cell().array, assembledCell.getPBC())
        structure = self.structureType(atomTypes, positions, cell=cell)
        newSystem = disassembler.disassemble(structure)
        if 'adjustedStructure' in self.targetProperties:
            newSystem['adjustedMolecules'] = newSystem['molecules']
            newSystem['adjustedCell'] = newSystem['cell']
            newSystem['adjustedEnvironment'] = newSystem['environment']
            del newSystem['molecules']
            del newSystem['cell']
            del newSystem['environment']
        system.update(**newSystem)

    def readPressureTensor(self, content, index=-1):
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

    def readDielectricProperties(self, content):
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

    def readDipoleMoment(self, content):
        dipoleMoment = np.zeros([1, 3])
        for line in content:
            if 'dipolmoment' in line:
                dipoleMoment = np.array(line.split()[1:4], dtype=float)
        return dipoleMoment

    def readFermi(self, content):
        energyFermi = None
        for line in content:
            if 'E-fermi' in line:
                energyFermi = float(line.split()[2])
        return energyFermi

    def readElasticMatrix(self, content):
        elasticMatrix = np.zeros((6, 6), dtype=float)
        for i, line in enumerate(content):
            if 'TOTAL ELASTIC MODULI' in line:
                for j, row in enumerate(content[i + 3: i + 9]):
                    elasticMatrix[j, :] = np.array(row.split()[1: 7], dtype=float)
        return elasticMatrix


    @classmethod
    def registerTypes(cls, structureType, atomType, cellType, atomicDisassemblerType):
        cls.structureType = structureType
        cls.atomType = atomType
        cls.cellType = cellType
        cls.atomicDisassemblerType = atomicDisassemblerType
