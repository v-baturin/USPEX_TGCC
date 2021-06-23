'''
@file        LAMMPS_Calculator.py
@author:     Arslan Mazitov
@copyright:  2019 Oganov's Lab. All rights reserved.
@contact:    arslan.mazitov@phystech.edu
@date        10 Jan 2019
@brief       Class for calculator of LAMMPS
'''

__author__ = 'a.mazitov'

import logging
import os
import shutil

import numpy as np
from ase.io import read
from typing import List
from os.path import join as pj
from scipy.spatial.transform.rotation import Rotation

from ..Atomistic.Transformation import Transformation
from .Common.SHELL_Interface import SHELL_Interface
logger = logging.getLogger(__name__)



class LAMMPS_Interface(SHELL_Interface):
    '''
    Calculator for LAMMPS.
    Local running
    '''

    sleepTime = 10
    _required_thermo_style_properties = ['ke', 'pe', 'etotal', 'enthalpy', 'pxx', 'pyy','pzz', 'pxy', 'pyz', 'pxz']
    _required_dump_properties = ['vx', 'vy', 'vz', 'fx', 'fy', 'fz']
    failedSystems = []

    inputFile = 'lammps.in'
    outputFile = 'lammps.out'
    errorFile = 'lammps.err'
    logFile = 'log.lammps'
    
    def __init__(self, tag: str, lammps_in: str = None, libs:List[str] = None, vacuumSize=10, **kwargs):
        '''

        :param params: dictionary with parameters:
                * lammps_in: (str) path to lammps.in-file.
                * libs: (list) list of paths to interatomic potentials.
        '''

        super().__init__(**kwargs)
        if lammps_in is None:
            lammps_in = pj(os.getcwd(), f'Specific/lammps.in_{tag}')

        assert os.path.exists(lammps_in)

        self.lammps_in  = lammps_in
        self.libs = libs if libs else []
        self.vacuumSize = vacuumSize
        logger.debug('LAMMPS calculator created.')

    def prepareLocalCalculation(self, system, calcFolder : str):
        '''
        :param system:
        :param calcFolder:
        '''
        if not os.path.exists(calcFolder):
            os.makedirs(calcFolder)
        # shutil.copy2(self.lammps_in, pj(calcFolder, self.inputFile))
        
        with open(self.lammps_in, 'r') as f:
            content = f.readlines()

        if not np.any(['variable f string lammps.dump' in item for item in content]):
            content.insert(0, 'variable f string lammps.dump\n')

        try:
            index = np.flatnonzero(['read_data' in item for item in content])[0]
            content[index] = 'read_data STRUC\n'
        except IndexError:
            index = np.flatnonzero(['box' in item for item in content])[0]
            content.insert(index+1, 'read_data STRUC\n')

            
        # index = np.flatnonzero(['pair_coeff' in item for item in content])[0]
        # composition = list(system.composition.keys())
        # key = self.libs[0].split('/')[-1]
        # key_idx = content[index].split().index(key)+1
        # pair_coeff = content[index].split()[:key_idx] + composition
        # # if pair_coeff[3] != self.libs[0]:
        # #     pair_coeff[3] = self.libs[0]
        # content[index] = " ".join(pair_coeff)
        # content[index] += '\n'
            
        # index = np.flatnonzero(['thermo_style' in item for item in content])[0]
        # thermo_style = content[index].splitlines()[0]
        # for prop in self._required_thermo_style_properties:
        #     if prop not in thermo_style:
        #         thermo_style += " " + prop
        # thermo_style += '\n'
        # content[index] = thermo_style

        try:
            index = np.flatnonzero(['run ' in item for item in content])[-1]
            nsteps = int(content[index].split()[-1])
        except:
            nsteps = 1000

        dump = f'dump d1 all custom {nsteps} ${{f}} id type x y z vx vy vz fx fy fz\n'
        try:
            index = np.flatnonzero(['dump ' in item for item in content])[-1]
            # dump = content[index].splitlines()[0]
            # dump_name = dump.split()[5]
            # if dump_name == 'lammps.dump' or dump_name == '${f}':
            #     pass
            # else:
            #     dump = content[index].split()
            #     dump[5] = 'lammps.dump'
            #     dump = ' '.join(dump)
            #
            # for prop in self._required_dump_properties:
            #     if prop not in dump:
            #         dump += " " + prop
            #
            # dump += '\n'
            content[index] = dump
        except:
            index = np.flatnonzero(['run' in item for item in content])[-1]
            content.insert(index, dump)

        for i in reversed(np.flatnonzero(['dump_modify' in item for item in content])):
            del content[i]

        
        with open(pj(calcFolder, self.inputFile), 'w') as f:
            f.writelines(content)
             
        write_data(system, pj(calcFolder, 'STRUC'), self.vacuumSize)
        for lib in self.libs:
            if isinstance(lib,str) and os.path.exists(lib):
                shutil.copy(lib,calcFolder)
        logger.debug('LAMMPS calculator prepared calculation.')

    def isConverged(self, calcFolder : str):
        lammps_completed = False
        tolerance_achieved = False
        if os.path.exists(pj(calcFolder, self.outputFile)):
            output = pj(calcFolder, self.outputFile)
        elif os.path.exists(pj(calcFolder, self.logFile)):
            output = pj(calcFolder, self.logFile)
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
            if 'No selection data provided with MLIP' in line:
                lammps_completed = True
                tolerance_achieved = True
                struct = read(pj(calcFolder, 'STRUC'), format='lammps-data', style='atomic')
                write_cfg(pj(calcFolder, 'sampled.cfg'), struct)

        
        if not tolerance_achieved:
            logger.error('LAMMPS minimization tolerance criteria is not achieved.')
            shutil.copy(output,  pj(calcFolder, 'ERROR'+self.outputFile))
            self.failedSystems.append(calcFolder)
        return lammps_completed and tolerance_achieved

    def readOutput(self, system, calcFolder : str):
        disassembler = system['disassembler']
        del system['disassembler']
        structure = system['structure']
        del system['structure']

        from ase.io.lammpsrun import read_lammps_dump
        atoms = read_lammps_dump(pj(calcFolder, 'lammps.dump'))
        # TODO appears to be buggy
        # cdisp = []
        # with open(pj(calcFolder, 'lammps.dump'), 'r') as g:
        #     glines = g.readlines()
        # for line in glines[5:8]:
        #     cdisp.append(-float(line.split()[0]))
        # atoms.translate(cdisp)

        cell = structure.getCell()
        system.update(disassembler.disassemble(type(structure)(structure.getAtomTypes(), atoms.get_positions(),
                                                               cell=type(cell)(atoms.get_cell().array, cell.getPBC()))))

        properties = self.readProperties(calcFolder)
        system['enthalpy'] = properties['TotEng']
        # system['stress'] = properties['pressureTensor']
        system['energy']= properties['TotEng']

    def readProperties(self, calcFolder: str):
        with open(pj(calcFolder, self.outputFile), 'r') as f:
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
        # pressureTensor = np.zeros((3,3))
        # pressureTensor[0][0], pressureTensor[1][1], pressureTensor[2][2] = properties['Pxx'], properties['Pyy'], properties['Pzz']
        # pressureTensor[0][1] = properties['Pxy']
        # pressureTensor[1][0] = properties['Pxy']
        # pressureTensor[0][2] = properties['Pxz']
        # pressureTensor[2][0] = properties['Pxz']
        # pressureTensor[1][2] = properties['Pyz']
        # pressureTensor[2][1] = properties['Pyz']
        # properties['pressureTensor'] = pressureTensor
        return properties

def write_cfg(fname, item):
    from ase.calculators.calculator import Calculator
    with open(fname, 'w') as f:
        forces = None
        stresses = None
        energy = None
        positions = item.get_positions()
        if hasattr(item, '_calc'):
            if isinstance(item._calc, Calculator):
                results = item._calc.results
                if 'forces' in results.keys():
                    forces = results['forces']
                if 'stress' in results.keys():
                    stresses = results['stress']
                if 'energy' in results.keys():
                    energy = results['free_energy']

        f.write("BEGIN_CFG\n")
        f.write(" Size\n")
        f.write(f" {len(item)}\n")
        f.write(" Supercell\n")
        for vector in item.get_cell():
            f.write(f"\t{vector[0]:.6f}\t{vector[1]:.6f}\t{vector[2]:.6f}\n")
        atom_header = " AtomData:  id type       cartes_x      cartes_y      cartes_z           "
        if forces is not None:
            atom_header += "fx          fy          fz"
        atom_header += "\n"
        f.write(atom_header)
        for i, atom in enumerate(item):
            p = positions[i]
            atom_data = f"  \t  {i+1}\t{item.numbers[i]-1}\t{p[0]:.6f}\t\t{p[1]:.6f}\t\t{p[2]:.6f}\t\t"
            if forces is not None:
                fs = forces[i]
                atom_data += f"{fs[0]:.6f}\t{fs[1]:.6f}\t{fs[2]:.6f}"
            atom_data += "\n"
            f.write(atom_data)
        if energy is not None:
            f.write(" Energy\n")
            f.write(f"\t{energy:.12f}\n")
        if stresses is not None:
            f.write(" PlusStress:  xx          yy          zz          yz          xz          xy\n")
            f.write(f"\t{stresses[0]:.5f}\t{stresses[1]:.5f}\t{stresses[2]:.5f}\t{stresses[3]:.5f}\t{stresses[4]:.5f}\t{stresses[5]:.5f}\n")
        if energy is not None or forces is not None or stresses is not None:
            f.write("Feature   EFS_by    VASP\n")
        if hasattr(item, 'ID'):
            f.write(f"Feature   AS_index {item.ID}\n")
        f.write("END_CFG\n")
        f.write("\n")
    
def write_data(system, filename, vacuumSize, comment=None):
    molecules = system['molecules']
    cell = system['cell']
    systemFactory = type(molecules[0])
    structure, disassembler = systemFactory.assemble(molecules, cell=cell)
    coordinates = structure.getCartesianCoordinates()
    cell = structure.getRectifiedCell().getEnvelopeCell(coordinates, vacuumSize)
    coordinates = cell.center(coordinates)
    structure = systemFactory(structure.getAtomTypes(), coordinates, cell)
    system['structure'] = structure
    system['disassembler'] = disassembler

    from ase.data import atomic_masses, atomic_numbers
    a = cell.getCellVectors()[0]
    rotation, _ = Rotation.align_vectors(a.reshape(1,3), np.array([[1.0, 0.0, 0.0]]))
    transformation = Transformation.fromRotVector(rotation.as_rotvec(), [0.,0.,0.])
    structure = transformation.transform(structure)
    cell = structure.getCell().getCellVectors()
    # specorder = system.specorder
    specorder, atom_ids = np.unique([el.short_name for el in structure.getAtomTypes()], return_inverse=True)
    positions = structure.getCartesianCoordinates()
    num_unique_types = len(specorder)
    # atom_ids = [specorder.index(item)+1 for item in system.get_chemical_symbols()]
    masses = [atomic_masses[atomic_numbers[item]] for item in specorder]

    with open(filename, 'w') as f:
        if comment is None:
            comment = 'LAMMPS_Interface data file'
        f.write(comment.strip() + '\n\n')

        f.write('{} atoms\n'.format(len(structure)))
        f.write('{} atom types\n'.format(num_unique_types))
        f.write('\n')
        f.write('{0:16.8e} {1:16.8e} xlo xhi\n'.format(0.0, cell[0, 0]))
        if cell[1,1] < 0:
            f.write('{0:16.8e} {1:16.8e} ylo yhi\n'.format(cell[1, 1], 0.0))
        else:
            f.write('{0:16.8e} {1:16.8e} ylo yhi\n'.format(0.0, cell[1, 1]))

        if cell[2, 2] < 0:
            f.write('{0:16.8e} {1:16.8e} zlo zhi\n'.format(cell[2, 2], 0.0))
        else:
            f.write('{0:16.8e} {1:16.8e} zlo zhi\n'.format(0.0, cell[2, 2]))
        f.write('{0:16.8e} {1:16.8e} {2:16.8e} xy xz yz\n'
                ''.format(cell[1, 0], cell[2, 0], cell[2, 1]))

        f.write('\nMasses\n\n')
        for id, mass in enumerate(masses):
            f.write('{} {}\n'.format(id+1, mass))
        f.write('\nAtoms # atomic\n\n')
        for i, (id, pos) in enumerate(
                zip(atom_ids+1, positions)):
            f.write('{} {}{:16.8e} {:16.8e} {:16.8e}\n'
                    .format(i + 1, id, pos[0], pos[1], pos[2]))
