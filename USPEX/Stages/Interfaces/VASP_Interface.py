"""
USPEX.Stages.VASP_Interface
===========================

"""

import logging
import numpy as np
import shutil
from pathlib import Path
from typing import List

from .KPoints import KPoints, BadKPoints

logger = logging.getLogger(__name__)


def split_up_data(data: List[str], out_size:int):
def split_up_data(data: List[str], out_size:int):
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

    # working input files
    incar_file = 'INCAR'
    kpoints_file = 'KPOINTS'
    poscar_file = 'POSCAR'
    potcar_file = 'POTCAR'


    DEFAULT_SLEEP_TIME = 30

    aseAdapterType = None

    @classmethod
    def registerTypes(cls, aseAdapterType):
        cls.aseAdapterType = aseAdapterType

    def __init__(self, tag: str,
                       kresol: float,
                       incar: str = None,
                       potcarsPath: str = None,
                       targetProperties: list = None,
                       **kwargs):
        '''

        :param tag:
        :param kresol:
        :param incar:
        :param potcarsPath:
        :param perturbate:
        :param vacuumSize:
        :param targetProperties:
        :param kwargs:
        '''

        self.incar = Path(incar) if incar is not None else Path.cwd()/f'Specific/INCAR_{tag}'
        assert self.incar.exists()

        self.potcarsPath = Path(potcarsPath) if potcarsPath is not None else Path.cwd()/'Specific'
        assert self.potcarsPath.exists()

        self.adapter = self.aseAdapterType()
        self.kPoints = KPoints(kresol)
        self.failedSystems = []

        self.targetProperties = targetProperties if targetProperties is not None else ['structure', 'enthalpy']

    def prepareLocalCalculation(self, system, calcFolder: Path):
    def prepareLocalCalculation(self, system, calcFolder: Path):
        '''
        :param system: our system
        :param calcFolder: calculation folder
        :return:
        '''
        with open(calcFolder/self.inputFile, 'wt') as f:
        with open(calcFolder/self.inputFile, 'wt') as f:
            pass

        structure = system['structure']

        ############################# POSCAR ##################################

        system['ase'] = self.adapter.write(structure, system['disassembler'].fixedIndices, f"EA{system['ID']}",
                                           calcFolder)

        ############################## INCAR ##################################
        shutil.copy2(self.incar, calcFolder/self.incar_file)
        shutil.copy2(self.incar, calcFolder/self.incar_file)

        if system['externalPressure']:
            with open(calcFolder/self.incar_file, 'a') as myfile:
            with open(calcFolder/self.incar_file, 'a') as myfile:
                myfile.write(f"\nPSTRESS={10 * system['externalPressure']:10f}\n")
        if calcFolder in self.failedSystems:
            with open(calcFolder/self.incar_file, 'a') as myfile:
            with open(calcFolder/self.incar_file, 'a') as myfile:
                myfile.write('ISYM=0\n')

        ############################# POTCAR ##################################

        potcar = calcFolder/'POTCAR'
        potcar.unlink(missing_ok=True)
        with open(potcar, 'w') as outfile:
            for atomType in np.unique([el.short_name for el in structure.getAtomTypes()]):
                with open(self.potcarsPath/f'POTCAR_{atomType}') as infile:
                    outfile.write(infile.read())

        ############################# KPOINTS #################################

        try:
            kPoints = self.kPoints.build(structure.getCell())
        except BadKPoints:
            # This LATTICE is extremely wrong, let's skip it from now
            logger.info('K-points cannot be built, so it\'s set as   [1, 1, 1]')
            kPoints = [1, 1, 1]

        with open(calcFolder/self.kpoints_file, 'w') as fp:
        with open(calcFolder/self.kpoints_file, 'w') as fp:
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

        return ''

    def isConverged(self, calcFolder: Path):
        '''
        :param calcFolder:
        :return: (bool) whether system calculation converged
        '''
        calcFolder = Path(calcFolder)

        if not (calcFolder.joinpath(self.outcar_file).exists() and
                calcFolder.joinpath(self.oszicar_file).exists() and
                calcFolder.joinpath(self.contcar_file).exists()):
            return False

        # Checking whether converge
        NELM = -1
        row_number = None
        with open(calcFolder/self.oszicar_file, 'r') as f:
            content = f.readlines()
            for i in range(len(content)):
                if content[i].find(' F= ') >= 0:
                    row_number = i
            if row_number is None:
                return False

            # Read previous line to check the number of SCF steps:
            vaspSCFsteps = int(content[row_number - 1].split(':')[1].split()[0].strip())

        with open(calcFolder/self.outcar_file, 'r') as f:
            for line in f:
                if line.find(' NELM ') >= 0:
                    NELM = int(line.split(' = ')[1].split()[0].replace(';', '').strip())
                    break

        # The calculation is considered successful in case vaspSCFsteps < NELM:
        if vaspSCFsteps < NELM:
            return True
        else:
            logger.error('VASP SCF is not converged.')
            shutil.copy2(calcFolder/self.outcar_file, calcFolder/f'ERROR-{self.outcar_file}')
            self.failedSystems.append(calcFolder)
            return False

    ############reading part

    def readOutput(self, system, calcFolder: Path):
        trajectory = self.adapter.read(calcFolder, **system.pop('ase'))
        aseResults = trajectory[-1]['results']
        results = {}
        if 'structure' in self.targetProperties:
            results['structure'] = trajectory[-1]['structure']
        if 'enthalpy' in self.targetProperties:
            results['enthalpy'] = aseResults.getEnthalpy(system['externalPressure'])
        if 'energy' in self.targetProperties:
            results['energy'] = aseResults.results['energy']
        if 'forces' in self.targetProperties:
            results['forces'] = aseResults.results['forces']
        if 'trajectory' in self.targetProperties:
            for subsystem in trajectory:
                subsystem['disassembler'] = system['disassembler']
                subsystem['externalPressure'] = system['externalPressure']
            results['trajectory'] = trajectory

        with open(calcFolder/self.outcar_file, 'rt') as fp:
        with open(calcFolder/self.outcar_file, 'rt') as fp:
            content = fp.readlines()
        if 'stressTensor' in self.targetProperties:
            results['stressTensor'] = self.readPressureTensor(content)
        if 'dielectricTensor' in self.targetProperties:
            results['dielectricTensor'] = self.readDielectricProperties(content)
        if 'dipoleMoment' in self.targetProperties:
            results['dipoleMoment'] = self.readDipoleMoment(content)
        if 'energyFermi' in self.targetProperties:
            results['energyFermi'] = self.readFermi(content)
        if 'elasticConstants' in self.targetProperties:
            results['elasticConstants'] = self.readElasticMatrix(content)
        return results

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
