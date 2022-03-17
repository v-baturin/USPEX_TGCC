"""
USPEX.Calculators.LAMMPS_Interface
==================================

.. codeauthor:: Arslan Mazitov <arslan.mazitov@phystech.edu>

"""
import logging
import os
import shutil

import numpy as np
from ase.io import read
from ase import Atoms
from typing import List
from os.path import join as pj
from .Common.SHELL_Interface import SHELL_Interface
logger = logging.getLogger(__name__)


AVAILABLE_TARGET_OBEJECTS = ['default', 'environment']
REQUIRED_THERMO_STYLE_PROPERTIES = ['enthalpy', 'etotal', 'ke', 'pe', 'temp', 
                                    'pxx', 'pyy', 'pzz', 'pxy', 'pxz', 'pyz']

class LAMMPS_Interface(SHELL_Interface):
    '''
    Calculator for LAMMPS.
    Local running
    '''

    # working output files
    inputFile = 'lammps.in'
    outputFile = 'lammps.out'
    
    log_file = 'log.lammps'
    data_file = 'STRUC'
    dump_file = 'lammps.dump'
    


    _DEFAULT_SLEEP_TIME = 30
    structureType = None
    atomType = None
    cellType = None
    atomicDisassemblerType = None

    def __init__(self, tag: str, lammps_in: str, libs: List[str], specorder: List[str],
                 vacuumSize: float = 10.0, targetObject: str = 'default', **kwargs):
        '''

        :param params: dictionary with parameters:
                * lammps_in: (str) path to lammps.in-file.
                * libs: (list) list of paths to interatomic potentials and associated files.
        '''

        super().__init__(**kwargs)

        self.lammps_in = lammps_in
        self.specorder = specorder
        assert os.path.exists(self.lammps_in)

        if libs is not None:
            self.libs = libs

        assert all([os.path.exists(lib) for lib in libs])

        self.failedSystems = []
        self.vacuumSize = vacuumSize
        self.targetObject = targetObject.lower()
        assert self.targetObject in AVAILABLE_TARGET_OBEJECTS

    def prepareLocalCalculation(self, system, calcFolder : str):
        '''
        :param system:
        :param calcFolder:
        '''
        structure, disassembler = self.structureType.assemble(**system)
        system['disassembler'] = disassembler

        coordinates = structure.getCartesianCoordinates()
        cell = structure.getRectifiedCell().getEnvelopeCell(coordinates, self.vacuumSize)
        coordinates = cell.center(coordinates)
        
        if not os.path.exists(calcFolder):
            os.makedirs(calcFolder)
        
        with open(self.lammps_in, 'r') as f:
            content = f.readlines()
        
        
        # Step 1. We check if our lammps data file will be read properly 
        # with read_data command when calculation is started
        
        read_data_occurence = ['read_data' in line for line in content]
        pair_style_occurence = ['pair_style' in line for line in content]
        read_data_content = f'read_data {self.data_file}\n'
        if any(read_data_occurence):
            index = np.where(read_data_occurence)[0][0]
            content[index] = read_data_content
        elif any(pair_style_occurence):
            index = np.where(pair_style_occurence)[0][0]
            content.insert(index-1, read_data_content)
            
        # Step 2. We check if all required properties are specified with thermo_style
        # command for further consistent output data reading 
        
        thermo_style_occurence = ['thermo_style' in line for line in content]
        thermo_style_content = 'thermo_style custom step ' + ' '.join(REQUIRED_THERMO_STYLE_PROPERTIES) + '\n'
        if any(thermo_style_occurence):
            index = np.where(read_data_occurence)[0][0]
            content[index] = thermo_style_content
        else:
            index = np.where(pair_style_occurence)[0][0]
            content.insert(index+1, thermo_style_content)
        # Step 3. We check if lammps.dump file will be written properly 
        # with dump command after calculation is finished
        
        lammps_uspex_dump = f'dump lammps_uspex_dump all custom 1 lammps.dump id type x y z vx vy vz fx fy fz\n'
        content.append(lammps_uspex_dump)
        content.append('run 0\n')
    
        # Step 4. We write all the input files to our calcFolder
        
        with open(pj(calcFolder, self.inputFile), 'w') as f:
            f.writelines(content)
        
        if self.targetObject == 'default':
            atoms = Atoms([el.short_name for el in structure.getAtomTypes()], coordinates, cell = cell.getCellVectors())
            atoms.write(pj(calcFolder, self.data_file), format='lammps-data', specorder=self.specorder)
        elif self.targetObject == 'environment':
            environment = system['environment'].getStructure()
            atoms = Atoms([el.short_name for el in environment.getAtomTypes()], environment.getCartesianCoordinates(), cell = cell.getCellVectors())
            atoms.write(pj(calcFolder, self.data_file), format='lammps-data',  specorder=self.specorder)
        
        for lib in self.libs:
            shutil.copy2(lib, calcFolder)
                

    def isConverged(self, calcFolder : str):
        lammps_completed = False
        tolerance_achieved = False
        if os.path.exists(pj(calcFolder, self.outputFile)):
            output = pj(calcFolder, self.outputFile)
        elif os.path.exists(pj(calcFolder, self.log_file)):
            output = pj(calcFolder, self.log_file)
        else:
            return False
        
        with open(output, 'r') as f:
            content = f.readlines()

        for line in reversed(content):
            if 'Total wall time' in line:
                lammps_completed = True
            if 'Stopping criterion' in line: 
                if 'energy tolerance' in line or 'force tolerance' in line or 'quadratic' in line:
                    tolerance_achieved = True
                if 'linealpha' in line:
                    tolerance_achieved = False
            if 'Verlet run' in line:
                tolerance_achieved = True
            if 'Breaking threshold exceeded' in line:
                lammps_completed = True
                tolerance_achieved = True
        
        if not tolerance_achieved:
            logger.error('LAMMPS minimization tolerance criteria is not achieved.')
            shutil.copy(output,  pj(calcFolder, 'ERROR-'+self.outputFile))
            self.failedSystems.append(calcFolder)
        return lammps_completed and tolerance_achieved        

    def readOutput(self, system, calcFolder : str):
        cell = system['cell']
        disassembler = system['disassembler']
        del system['disassembler']

        properties = self.readProperties(calcFolder)
        
        if self.targetObject == 'default':
            atoms = read(pj(calcFolder, self.dump_file), format='lammps-dump-text')
            positions = atoms.get_positions()
            numbers = atoms.get_atomic_numbers()
            symbols = [self.specorder[i-1] for i in numbers]
            atomTypes = np.array([self.atomType(symbol) for symbol in symbols], dtype=self.atomType)
            cell = self.cellType(atoms.get_cell().array, cell.getPBC()).getEnvelopeCell(positions, 0)
            positions = cell.center(positions)
            system.update(disassembler.disassemble(self.structureType(atomTypes, positions, cell=cell)))
            system['enthalpy'] = properties['Enthalpy']
            system['energy'] = properties['TotEng']
            system['stressTensor'] = properties['StressTensor']
        elif self.targetObject == 'environment':
            system['environmentEnthalpy'] = properties['Enthalpy']

    def readProperties(self, calcFolder: str):
        if os.path.exists(pj(calcFolder, self.outputFile)):
            output = pj(calcFolder, self.outputFile)
        elif os.path.exists(pj(calcFolder, self.log_file)):
            output = pj(calcFolder, self.log_file)
        else:
            raise FileNotFoundError('Cannot find either {self.outputFile} or {self.log_file}.')
        with open(output, 'r') as f:
            content = f.readlines()
        
        for i, line in enumerate(content):
            if "Step" in line:
                properties_list = line.split()
                properties = dict().fromkeys(properties_list)
            if "Loop" in line:
                end_ind = i-1
        templine = content[end_ind].split()
        for j, item in enumerate(templine):
            if properties_list[j] == "Step":
                properties[properties_list[j]] = int(item)
            else:
                properties[properties_list[j]] = float(item)
        stressTensor = np.zeros((3,3))
        stressTensor[0][0] = properties['Pxx']
        stressTensor[1][1] = properties['Pyy']
        stressTensor[2][2] = properties['Pzz']
        stressTensor[0][1] = stressTensor[1][0] = properties['Pxy']
        stressTensor[0][2] = stressTensor[2][0] = properties['Pxz']
        stressTensor[1][2] = stressTensor[2][1] = properties['Pyz']
        properties['StressTensor'] = stressTensor
        return properties

    @classmethod
    def registerTypes(cls, structureType, atomType, cellType, atomicDisassemblerType):
        cls.structureType = structureType
        cls.atomType = atomType
        cls.cellType = cellType
        cls.atomicDisassemblerType = atomicDisassemblerType

