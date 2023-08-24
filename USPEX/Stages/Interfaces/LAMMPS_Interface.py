"""
USPEX.Stages.LAMMPS_Interface
=============================

.. codeauthor:: Arslan Mazitov <arslan.mazitov@phystech.edu>

"""
import logging
import numpy as np
import shutil

from pathlib import Path
from typing import List
from ase.io import read


logger = logging.getLogger(__name__)

REQUIRED_THERMO_STYLE_PROPERTIES = ['enthalpy', 'etotal', 'ke', 'pe', 'temp', 'pxx', 'pyy', 'pzz', 'pxy', 'pxz', 'pyz']


class LAMMPS_Interface:
    """
    Calculator for LAMMPS.
    Local running
    """

    # working output files
    inputFile = 'lammps.in'
    outputFile = 'lammps.out'
    errorFile = 'error'

    log_file = 'log.lammps'
    data_file = 'STRUC'
    dump_file = 'lammps.dump'
    mlip_ini = 'mlip.ini'
    mlip_in = None
    mlip_sample = None
    mlip_mtp = 'default.mtp'
    
    DEFAULT_SLEEP_TIME = 30

    AtomicStructureRepresentation = None

    @classmethod
    def registerTypes(cls, AtomicStructureRepresentation):
        cls.AtomicStructureRepresentation = AtomicStructureRepresentation

    def __init__(self, tag: str, specorder: List[str], lammps_in: str = None, mlip_in: str = None, mlip: str = None,
                 libs: List[str] = None, targetProperties: list = None, **kwargs):
        """

        :param params: dictionary with parameters:
                * lammps_in: (str) path to lammps.in-file.
                * libs: (list) list of paths to interatomic potentials and associated files.
        """

        self.tag = tag
        self.lammps_in = Path.cwd()/f'Specific/lammps.in_{tag}' if lammps_in is None else Path(lammps_in)
        assert self.lammps_in.exists()

        self.mlip = mlip
        if self.mlip is not None:
            self.mlip_in = Path.cwd()/f'Specific/mlip.ini_{tag}' if mlip_in is None else Path(mlip_in)
            assert self.mlip_in.exists()

            with open(self.mlip_in, 'r') as f:
                content = f.readlines()

            for line in content:
                if 'sample:save_sampled_to' in line:
                    self.mlip_sample = line.split()[1]

            for line in content:
                if 'mlip:load_from' in line:
                    self.mlip_mtp = line.split()[1]
                    break
            else:
                raise RuntimeError('Bad mlip.ini: load_from not specified.')

        self.specorder = specorder

        self.libs = [] if libs is None else [Path(lib) for lib in libs]
        assert all([lib.exists() for lib in self.libs])

        self.failedSystems = []
        self.targetProperties = targetProperties if targetProperties is not None else ['structure', 'enthalpy']

    def prepareLocalCalculation(self, system, calcFolder: Path):
        """
        :param system:
        :param calcFolder:
        """

        structure = system.getProperty('structure', extension='atomistic', suffix='intermediate')
        cell = structure.getCell()
        with open(calcFolder/'pbc', 'wt') as f:
            f.write(' '.join(f'{c}' for c in cell.getPBC()))

        atoms = self.AtomicStructureRepresentation.toAtoms(structure)
        filename = calcFolder / self.data_file
        atoms.write(filename, format='lammps-data', specorder=self.specorder)
        with open(filename, 'rt') as f:
            content = f.readlines()
        content[0] = f"EA{system['ID']}\n"
        with open(filename, 'wt') as f:
            f.writelines(content)

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
            index = np.where(thermo_style_occurence)[0][0]
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
        
        with open(calcFolder/self.inputFile, 'w') as f:
            f.writelines(content)

        for lib in self.libs:
            shutil.copy2(lib, calcFolder)

        if self.mlip is not None:
            shutil.copy2(self.mlip, calcFolder/self.mlip_mtp)
            shutil.copy2(self.mlip_in, calcFolder/self.mlip_ini)

        return ''

    def isConverged(self, calcFolder: Path):
        lammps_completed = False
        tolerance_achieved = False
        if calcFolder.joinpath(self.outputFile).exists():
            output = calcFolder/self.outputFile
        elif calcFolder.joinpath(self.log_file).exists():
            output = calcFolder/self.log_file
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
            shutil.copy(output,  calcFolder/f'ERROR-{self.outputFile}')
            self.failedSystems.append(calcFolder)
        return True        

    def readOutput(self, system, calcFolder: Path):
        with open(calcFolder / 'pbc', 'rt') as f:
            pbc = tuple(int(c) for c in f.read().split())
        if calcFolder.joinpath(self.dump_file).exists():
            atoms = read(calcFolder / self.dump_file, format='lammps-dump-text')
            atoms.set_pbc(pbc)
            atoms.set_chemical_symbols(self.specorder[i - 1] for i in atoms.get_atomic_numbers())
            structure = self.AtomicStructureRepresentation.fromAtoms(atoms)
            results = atoms.get_calculator().results
        else:
            results = None
            structure = None

        properties = self.readProperties(calcFolder)
        if 'structure' in self.targetProperties:
            system.setProperty('structure', structure, extension='atomistic', suffix=self.tag)
        if 'enthalpy' in self.targetProperties:
            if properties is not None:
                system.setProperty('enthalpy', properties['Enthalpy'], suffix=self.tag)
            elif results is not None:
                if 'enthalpy' in results:
                    system.setProperty('enthalpy', results['energy'], suffix=self.tag)
                else:
                    system.setProperty('energy', results['energy'], suffix=self.tag)
            else:
                raise RuntimeError("Bad lammps output.")
        if 'energy' in self.targetProperties:
            if properties is not None:
                system.setProperty('energy', properties['TotEng'])
            elif results is not None:
                system.setProperty('energy', results['energy'], suffix=self.tag)
            else:
                raise RuntimeError("Bad lammps output.")
        if 'forces' in self.targetProperties:
            if results is not None:
                system.setProperty('forces', results['forces'], suffix=self.tag)
            else:
                raise RuntimeError("Bad lammps output.")

        if 'stressTensor' in self.targetProperties:
            if properties is not None:
                stressTensor = np.zeros((3, 3))
                stressTensor[0][0] = properties['Pxx']
                stressTensor[1][1] = properties['Pyy']
                stressTensor[2][2] = properties['Pzz']
                stressTensor[0][1] = stressTensor[1][0] = properties['Pxy']
                stressTensor[0][2] = stressTensor[2][0] = properties['Pxz']
                stressTensor[1][2] = stressTensor[2][1] = properties['Pyz']
                system.setProperty('stressTensor', stressTensor, suffix=self.tag)
            else:
                raise RuntimeError("Bad lammps output.")

        if 'trajectory' in self.targetProperties:
            atomistic = system.flavourFactory.extensions['atomistic'].utility
            sample = atomistic.AtomicStructureRepresentation.readMLIPsample(calcFolder/self.mlip_sample, self.specorder)
            # for subsystem in sample:
            #     subsystem['disassembler'] = system['disassembler']
            #     subsystem['externalPressure'] = system['externalPressure']
            system.setProperty('trajectory', sample, suffix=self.tag)

        # TODO move to constraints
        # BAD_SYSTEM_ENERGY_PER_ATOM_THRESHOLD = 1e3
        # if abs(properties['TotEng']) / len(aseStructure) > BAD_SYSTEM_ENERGY_PER_ATOM_THRESHOLD:
        #     logger.error(f"System {system['ID']} seems to has wrong energy: {properties['TotEng']}. It will be discarded.")
        #     system['isBad'] = True

    def readProperties(self, calcFolder: Path):
        if calcFolder.joinpath(self.outputFile).exists():
            output = calcFolder/self.outputFile
        elif calcFolder.joinpath(self.log_file).exists():
            output = calcFolder/self.log_file
        else:
            raise FileNotFoundError(f'Cannot find either {self.outputFile} or {self.log_file}.')
        with open(output, 'r') as f:
            content = f.readlines()
        try:
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
        except Exception:
            properties = None
        return properties



