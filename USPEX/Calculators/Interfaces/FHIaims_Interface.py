"""
USPEX.Calculators.FHIaims_Interface
===================================

"""

import logging
import os
import shutil
import numpy as np
from os.path import join as pj

from USPEX.Calculators.Interfaces.KPoints import KPoints, BadKPoints

logger = logging.getLogger(__name__)


class FHIaims_Interface:

    DEFAULT_SLEEP_TIME = 30
    structureType = None
    atomType = None
    cellType = None
    atomicDisassemblerType = None

    inputFile, outputFile, errorFile = 'input', 'output', 'error'
    control_file = 'control.in'
    geometry_file = 'geometry.in'

    out_geometry_file = 'geometry.in.next_step'

    def __init__(self, tag: str, kresol: float, control: str = None, perturbate: bool = True, fixCell: bool = False,
                 vacuumSize=10, targetProperties: list = None, **kwargs):

        super().__init__(**kwargs)
        if control is None:
            control = pj(os.getcwd(), f'Specific/aims_control_{tag}')

        assert os.path.exists(control)

        with open(control, 'r') as f:
            self.control = f.read()

        self.kPoints = KPoints(kresol)

        self.perturbate = perturbate
        self.fixCell = fixCell
        self.vacuumSize = vacuumSize
        self.targetProperties = targetProperties if targetProperties is not None else ['structure', 'enthalpy']

    def prepareLocalCalculation(self, system, calcFolder : str):
        structure, disassembler = self.structureType.assemble(**system, vacuumSize=self.vacuumSize)
        system['disassembler'] = disassembler
        atomTypes = structure.getAtomTypes()
        system['symbolsOrder'] = np.argsort([el.short_name for el in atomTypes])
        cell = structure.getCell()
        system['assembledCell'] = cell
        coordinates = structure.getCartesianCoordinates()

        with open(pj(calcFolder, self.inputFile), 'wt') as f:
            pass

        with open(pj(calcFolder, self.control_file), 'wt') as dest:
            dest.write(self.control)

        try:
            kPoints = self.kPoints.build(cell)
        except BadKPoints:
            # This LATTICE is extremely wrong, let's skip it from now
            logger.info('K-points cannot be built, so it\'s set as   [1, 1, 1]')
            kPoints = [1, 1, 1]

        with open(pj(calcFolder, self.control_file), 'a') as f:
            f.write('k_grid {} {} {}'.format(*kPoints))

        with open(pj(calcFolder, self.geometry_file), 'wt') as fp:

            if cell.dim != 0:

                lat = cell.getCellVectors()
                fp.write('lattice_vector   {:12.6f} {:12.6f} {:12.6f}\n'.format(*lat[0, :]))
                fp.write('lattice_vector   {:12.6f} {:12.6f} {:12.6f}\n'.format(*lat[1, :]))
                fp.write('lattice_vector   {:12.6f} {:12.6f} {:12.6f}\n'.format(*lat[2, :]))
                if self.fixCell:
                    fp.write('constrain_relaxation .true.\n')

            fixedIndices = disassembler.envIndices[
                system['environment'].getFixedIndices()] if 'environment' in system else []
            for i, (symbol, coord) in enumerate(zip(structure.getAtomTypes(), coordinates)):
                fp.write('atom  {1:15.8f} {2:15.8f} {3:15.8f} {0:2s}\n'.format(symbol.short_name, *coord))
                if i in fixedIndices:
                    fp.write('constrain_relaxation .true.\n')

    def isConverged(self, calcFolder : str):
        if not os.path.exists(pj(calcFolder, self.outputFile)):
            return False
        with open(pj(calcFolder, self.outputFile), 'r') as f:
            content = f.read()
        if 'Have a nice day' not in content:
            logger.error('FHI-aims is not completely Done')
            return False
        if 'Total energy corrected' not in content:
            logger.error('FHI-aims is not done correctly!!!')
            logger.error('Read_FHI.aims : FHI 1st SCF is not correctly Done! ')
            return False
        return True

    def readOutput(self, system, calcFolder : str):

        # In FHI-081213 geometry.in.next_step automatically will be created but
        # for FHI-081219 user need to specify restart_relaxations .true.

        # This condition was applied because sometimes which systems is small
        # USPEX creates very good structures which are the same with the relaxed one
        # and FHI finishes without changing the relaxed structure, thus
        # geometry.in.next_step won't be created.

        if 'structure' in self.targetProperties:
            geometry_file = pj(calcFolder, self.out_geometry_file)
            if not os.path.exists(geometry_file):
                shutil.copy(pj(calcFolder, self.geometry_file), geometry_file)
            with open(geometry_file,'r') as f:
                content = f.read()
            self.readStructure(system, content)

        if 'enthalpy' in self.targetProperties:
            with open(pj(calcFolder, self.outputFile), 'r') as f:
                content = f.readlines()

            for line in content:
                if 'Total energy corrected' in line:
                    system['enthalpy'] = float(line.split()[5])
                    break

    def readStructure(self, system, content):
        content_list = content.split('\n')

        lattice = []
        coordinates = []
        atomTypes = []
        for line in content_list:
            if 'lattice_vector' in line:
                lattice.append([float(x) for x in line.split()[1:4]])
            if 'atom' in line:
                line = line.split()
                coordinates.append([float(x) for x in line[1:4]])
                atomTypes.append(self.atomType(line[4]))

        coor = np.array(coordinates)
        if 'lattice_vector' in content:
            lat = np.array(lattice)
        else:
            '''
            coor = bsxfun(@minus, coor, mean(coor)); %Vectorized
            lat_len1 = max(coor(:,1)) - min(coor(:,1)) + 10;
            lat_len2 = max(coor(:,2)) - min(coor(:,2)) + 10;
            lat_len3 = max(coor(:,3)) - min(coor(:,3)) + 10;
            lat = diag([lat_len1, lat_len2, lat_len3]);
            coor = bsxfun(@plus, coor, [lat_len1, lat_len2, lat_len3]/2);
            All this matlab code can be rewritten as simple as:
            '''
            coor -= coor.mean(axis=0)
            lat = np.diag(coor.max(axis=0) - coor.min(axis=0) + 10)
            coor += np.diag(lat * 0.5)

        assembledCell = system.pop('assembledCell')
        disassembler = system.pop('disassembler')
        cell = self.cellType(lat, assembledCell.getPBC())
        system.update(disassembler.disassemble(self.structureType(atomTypes, coor, cell=cell)))

    @classmethod
    def registerTypes(cls, structureType, atomType, cellType, atomicDisassemblerType):
        cls.structureType = structureType
        cls.atomType = atomType
        cls.cellType = cellType
        cls.atomicDisassemblerType = atomicDisassemblerType
