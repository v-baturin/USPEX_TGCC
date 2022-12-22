"""
USPEX.Stages.ABINIT_Interface
=============================

.. codeauthor:: Michele Galasso <m.galasso@yandex.com>

"""

import logging
import os
import shutil
import numpy as np
from os.path import join as pj
from typing import List

from .KPoints import KPoints, BadKPoints
from ...Presets import udateSystemWithPrefix as usp

logger = logging.getLogger(__name__)
EV_PER_CUBIC_ANGSTREM_PER_GPA = 1/160.21766208
GPA_TO_HARTREE_PER_CUBIC_BOHR = 1/29421.033


class ABINIT_Interface:
    """
    Calculator for ABINIT.
    Local running
    """

    # working output files
    out_file_name = 'abinit.out'
    gsr_file_name = 'abinit_o_GSR.nc'

    # working input files
    inputFile = 'abinit.files'
    in_file_name = 'abinit.in'
    outputFile = 'output'
    errorFile = 'error'


    DEFAULT_SLEEP_TIME = 30
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

    def __init__(self, tag: str, kresol: float,  in_file: str = None, pp_files: List[str] = None, perturbate: bool = True,
                 vacuumSize=10, targetProperties: list = None, environmentStyle=None, inStyle=None, **kwargs):
        """
        Initializes the class.

        :type in_file: str
        :param in_file: location of file .in, containing ABINIT calculation parameters.
        :type pp_files: list
        :param pp_files: location of pseudopotential files for all atomic types in the unit cell.
        :type kresol: float
        :param kresol: K-points resolution.
        :type kwargs: dict
        :param kwargs: parameters for initializing the parent class.
        """

        self.tag = tag
        self.tmp = f'tmp_{tag}'
        if in_file is None:
            in_file = pj(os.getcwd(), f'./Specific/abinit.in_{tag}')

        pp_files = pp_files if pp_files is not None else []

        assert os.path.exists(in_file)
        assert np.all([os.path.exists(pp_file) for pp_file in pp_files])

        self.in_file = in_file
        self.pp_files = pp_files

        self.kPoints = KPoints(kresol)
        self.failedSystems = []
        self.vacuumSize = vacuumSize
        self.targetProperties = targetProperties if targetProperties is not None else ['structure', 'enthalpy']
        self.perturbate = perturbate
        self.environmentStyle = environmentStyle
        self.inStyle = inStyle

    def prepareLocalCalculation(self, system, calcFolder : str):
        """
        :param system: our system
        :return:
        """
        structure, disassembler = self.atomicDisassemblerType.assemble(**system,
                                                                       style=self.environmentStyle,
                                                                       inStyle=self.inStyle,
                                                                       vacuumSize=self.vacuumSize)
        system[self.tmp]['disassembler'] = disassembler

        if self.perturbate:
            structure = structure.getPerturbatedStructure(disassembler.fixedIndices)

        cell = structure.getCell()
        system[self.tmp]['pbc'] = cell.getPBC()
        coordinates = structure.getCartesianCoordinates()
        atomTypes = structure.getAtomTypes()


        ############################# FILES FILE ################################
        for pp_file_path in self.pp_files:
            shutil.copy2(pp_file_path, calcFolder)

        with open(pj(calcFolder, self.inputFile), 'wt') as f:
            f.write(f'{self.in_file_name}\n'
                    f'{self.out_file_name}\n'
                    'abinit_i\n'
                    'abinit_o\n'
                    'abinit\n')

            # pp files need to be ordered by increasing atomic number
            pp_files_names = [os.path.split(pp_file_path)[1] for pp_file_path in self.pp_files]
            for pp_file_name in sorted(pp_files_names, key=lambda e: self.atomType(e.split('.')[0]).z):
                f.write(f'{pp_file_name}\n')

        ############################## IN FILE ##################################
        # the following parameters will be ignored since they are set by USPEX
        ignored_params = ['natom', 'acell', 'rprim', 'znucl', 'typat', 'ntypat', 'xred', 'xcart', 'xangst', 'strtarget']

        # here we will record the user-defined parameters
        user_params = []

        clean_in_file = ''
        with open(self.in_file, 'rt') as f:
            for line in f:
                clean_line = line.partition('#')[0]
                clean_line = clean_line.strip()

                param = clean_line.partition(' ')[0]
                if param in ignored_params:
                    msg = (f'The parameter {param:s} that you specified in {os.path.split(self.in_file)[1]:s}'
                           'will have no effect since it will be overwritten by USPEX.')
                    logger.warning(msg)
                else:
                    if param != '':
                        user_params.append(param)
                    clean_in_file += line

        with open(pj(calcFolder, self.in_file_name), 'wt') as f:
            f.write(clean_in_file)

        if system['externalPressure']:
            with open(pj(calcFolder, self.in_file_name), 'a') as myfile:
                abipressure = -1 * system['externalPressure'] * GPA_TO_HARTREE_PER_CUBIC_BOHR
                myfile.write(f'strtarget {abipressure:.2e} {abipressure:.2e} {abipressure:.2e} 0.0 0.0 0.0\n')
        if calcFolder in self.failedSystems:
            if 'kptopt' not in user_params:
                with open(pj(calcFolder, self.in_file_name), 'a') as myfile:
                    myfile.write('kptopt 2\n')

        # K-GRID
        try:
            kPoints = self.kPoints.build(cell)
        except BadKPoints:
            # This LATTICE is extremely wrong, let's skip it from now
            logger.info('K-points cannot be built, so it\'s set as   [1, 1, 1]')
            kPoints = [1, 1, 1]

        with open(pj(calcFolder, self.in_file_name), 'a') as f:
            f.write('\n# k-point grid\n')
            f.write('ngkpt   %d %d %d\n' % tuple(kPoints))
            f.write('nshiftk 1\n')
            f.write('shiftk  0 0 0\n')

        # DEFINITION OF THE ATOM TYPES AND UNIT CELL
        species = list(set(el.z for el in atomTypes))

        with open(pj(calcFolder, self.in_file_name), 'a') as f:
            f.write('\n# Definition of the unit cell\n')
            f.write('acell 1 1 1 angstrom\n')
            f.write('rprim\n')
            for v in cell.getCellVectors():
                f.write('%18.14f %18.14f %18.14f\n' % tuple(v))

            if 'chkprim' not in user_params:
                f.write('chkprim 0  # allow non-primitive cells\n')

            f.write('\n# Definition of the atom types\n')
            f.write('natom  %d\n' % (len(atomTypes)))
            f.write('ntypat %d\n' % (len(species)))
            f.write('znucl ')
            for Z in species:
                f.write(' %d' % Z)
            f.write('\n')

            f.write('\n# Enumerate different atomic species\n')
            f.write('typat\n')
            types = []
            for Z in [el.z for el in atomTypes]:
                for n, Zs in enumerate(species):
                    if Z == Zs:
                        types.append(n + 1)
            n_entries_int = 20  # integer entries per line
            for n, type_ in enumerate(types):
                f.write(' %d' % (type_))
                if n > 1 and ((n % n_entries_int) == 1):
                    f.write('\n')
            f.write('\n')

            f.write('\n# Definition of the atoms\n')
            f.write('xred\n')
            for pos in cell.cartesianToFractional(coordinates):
                f.write('%18.14f %18.14f %18.14f\n' % tuple(pos))

        return []

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

    def isConverged(self, calcFolder : str):
        """
        :param SYSTEM:
        :return: (bool) whether system calculation converged
        """

        if not (os.path.exists(pj(calcFolder, self.out_file_name))):
            return False

        if os.path.exists(pj(calcFolder, '__ABI_MPIABORTFILE__')):
            logger.error('ABINIT exited with error.')
            return False

        # Checking whether the SCF has converged
        with open(pj(calcFolder, self.out_file_name)) as f:
            for line in f:
                lowerline = line.lower()

                if lowerline.rfind('was not enough scf cycles to converge') > -1:
                    logger.error('ABINIT SCF is not converged.')
                    shutil.copy2(pj(calcFolder, self.out_file_name), f'{pj(calcFolder, "ERROR")}-{self.out_file_name}')
                    self.failedSystems.append(calcFolder)
                    return False

        return True

    def readOutput(self, system, calcFolder: str):
        if not os.path.isfile(pj(calcFolder, self.gsr_file_name)):
            msg = (f'file {self.gsr_file_name:s} not found in {os.path.basename(calcFolder):s}.'
                   'Your ABINIT executable needs to be compiled with NETCDF support in order to be used with USPEX.')
            raise IOError(msg)

        from abipy import abilab
        gsr = abilab.abiopen(pj(calcFolder, self.gsr_file_name))
        if 'structure' in self.targetProperties:
            structure = self.readStructure(gsr, system[self.tmp].pop('pbc'))
            usp(system, system[self.tmp].pop('disassembler').disassemble(structure), 'system', self.environmentStyle)
        if 'enthalpy' in self.targetProperties:
            enthalpy = float(gsr.energy) + np.linalg.det(gsr.structure.lattice.matrix) * system['externalPressure'] * \
                                 EV_PER_CUBIC_ANGSTREM_PER_GPA
            usp(system, enthalpy, 'enthalpy', self.environmentStyle)
        if 'forces' in self.targetProperties:
            usp(system, np.copy(gsr.cart_forces), 'forces', self.environmentStyle)
        if 'stressTensor' in self.targetProperties:
            usp(system, np.copy(gsr.cart_stress_tensor), 'stressTensor', self.environmentStyle)

    def readStructure(self, gsr, pbc):
        tmp_positions = gsr.structure.cart_coords
        atomTypes = [self.atomType(el.symbol) for el in gsr.structure.species]
        positions = np.empty(tmp_positions.shape, dtype=float)
        atomSymbols = [el.short_name for el in atomTypes]
        for i, position in zip(np.argsort(atomSymbols), tmp_positions):
            positions[i] = position
        cell = self.cellType(gsr.structure.lattice.matrix, pbc)
        return self.structureType(atomTypes, positions, cell=cell)
