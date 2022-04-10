"""
USPEX.Calculators.GULP_Interface
================================

"""

import logging
import numpy as np
import os
import re
import shutil
from os.path import join as pj
from typing import List

from .Common.SHELL_Interface import SHELL_Interface

logger = logging.getLogger(__name__)


class GULP_Interface(SHELL_Interface):
    """
    Calculator for Gulp.
    Local running
    """

    _DEFAULT_SLEEP_TIME = 10
    structureType = None
    atomType = None
    cellType = None
    atomicDisassemblerType = None

    def __init__(self, tag: str, ginput: str = None, goptions: str = None, libs: List[str] = None,
                 moleculeSpecifics: dict = None, perturbate: bool = True, fixCell: bool = False, vacuumSize = 10,
                 targetProperties: list = None, **kwargs):
        """

        :param params: dictionary with parameters:
                * ginput: (str) path to ginput-file.
                * goption: (str) path to goption-file.
                * commandExecutable: (str) executable command
                * remote: (dict) remote server params    # optional
                * taskManager: (dict) task managers params     # optional
        """

        super().__init__(**kwargs)
        if ginput is None:
            ginput = pj(os.getcwd(), f'Specific/ginput_{tag}')

        if goptions is None:
            goptions = pj(os.getcwd(), f'Specific/goptions_{tag}')

        self.optimizedStructure = 'optimized.structure'

        assert os.path.exists(ginput)
        assert os.path.exists(goptions)

        with open(ginput, 'r') as f:
            self.ginput = f.read()
        with open(goptions, 'r') as f:
            self.goptions = f.read()

        self.libs = libs if libs else []
        if moleculeSpecifics is not None:
            self.moleculeSpecifics = moleculeSpecifics
        else:
            self.moleculeSpecifics = {}

        self.perturbate = perturbate
        self.fixCell = fixCell
        self.vacuumSize = vacuumSize
        self.targetProperties = targetProperties if targetProperties is not None else ['structure', 'enthalpy']
        logger.debug('GULP calculator created.')

    def prepareLocalCalculation(self, system, calcFolder : str):
        """

        """

        structure, disassembler = self.structureType.assemble(**system)
        system['disassembler'] = disassembler

        coordinates = structure.getCartesianCoordinates()
        cell = structure.getRectifiedCell().getEnvelopeCell(coordinates, self.vacuumSize)
        system['assembledCell'] = cell
        coordinates = cell.center(coordinates)

        files_to_delete = ['output', 'optimized.structure']
        for f in files_to_delete:
            if os.path.isfile(f):
                os.remove(f)

        # TODO connectivities in molecular mode.
        # if system.isMolecule:
        #     # with open('ginput_{}'.format(step)) as f:
        #     #     content = f.read()
        #     # if not 'connect' in content:
        #     #     print('Connectivity is not specified by GULP, will be generated automatically')
        #     #     TO_write = True
        #     # else:
        #     #     TO_write = False
        #     #
        #     # content_to_write = Write_GULP_MOL(ORG_STRUC, POP_STRUC, Ind_No, TO_write)
        #     pass
        # else:

        lattice = type(cell)(cell.getCellVectors(), (1,1,1)).getCellParameters()

        content_to_write = ''
        content_to_write += 'cell\n'

        if self.fixCell:
            content_to_write += '%7.3f %7.3f %7.3f %7.3f %7.3f %7.3f 0 0 0 0 0 0 \n' % tuple(lattice)
        else:
            content_to_write += '%7.3f %7.3f %7.3f %7.3f %7.3f %7.3f\n' % tuple(lattice)

        content_to_write += 'fractional\n'

        if self.perturbate:
            coordinates += 0.1 * (np.random.rand(len(structure), 3) - 0.5)

        # TODO gulp specifics for atoms
        # symbols = []
        # for molSymbol, molecule in zip(system.molSymbol,system.molecules):
        #     if molSymbol in self.moleculeSpecifics:
        #         symbols.extend(self.moleculeSpecifics[molSymbol]['elementInLib'])
        #     else:
        #         symbols.extend(molecule.get_chemical_symbols())

        # TODO charges
        # if system.has('initial_charges') or system.has('charges'):
        #     for symbol, coord, charge in zip(symbols, system.get_scaled_positions(wrap = False), system.get_initial_charges()):
        #         tuple_to_format = tuple([symbol] + coord.tolist() + [charge])
        #         # if twoDimensional:
        #         #     # if POP_STRUC['POPULATION'][Ind_No]['chanAList'][coordLoop] == 1:
        #         #     #     content_to_write += '%4s %12.6f %12.6f %12.6f 1 1 0 1 1 1\n' % tuple_to_format
        #         #     # else:
        #         #     #     content_to_write += '%4s %12.6f %12.6f %12.6f 1 1 0 0 0 0\n' % tuple_to_format
        #         #     pass
        #         # else:
        #         #     content_to_write += '%4s %12.6f %12.6f %12.6f   core %12.6f\n' % tuple_to_format
        #         content_to_write += '%4s %12.6f %12.6f %12.6f   core %12.6f\n' % tuple_to_format
        # else:

        fixedIndices = disassembler.envIndices[system['environment'].getFixedIndices()] if 'environment' in system else []
        for i, (symbol, coord) in enumerate(zip(structure.getAtomTypes(), cell.cartesianToFractional(coordinates))):
            tuple_to_format = tuple([symbol.short_name] + coord.tolist())
            if cell.dim == 2:
                if i in fixedIndices:
                    content_to_write += '%4s %12.6f %12.6f %12.6f 1 1 0 1 1 1\n' % tuple_to_format
                else:
                    content_to_write += '%4s %12.6f %12.6f %12.6f 1 1 0 0 0 0\n' % tuple_to_format
            else:
                content_to_write += '%4s %12.6f %12.6f %12.6f\n' % tuple_to_format

        # Write part:
        total_content = self.goptions + '\n' + content_to_write + self.ginput + '\n'
        if system['externalPressure'] >= 0.05:
            total_content += f"pressure {system['externalPressure']:.1f}\n"
        total_content += 'dump every optimized.structure\n'

        with open(pj(calcFolder, self.inputFile), 'wt') as f:
            f.write(total_content)
        for lib in self.libs:
            if isinstance(lib,str) and os.path.exists(lib):
                shutil.copy(lib, calcFolder)

        logger.debug('GULP calculator prepared calculation.')

    def isConverged(self, calcFolder : str):
        """
        :param SYSTEM:
        :return: whether optimization converged
        """
        with open(pj(calcFolder, self.errorFile), 'rt') as fp:
            content = fp.readlines()
            if 'STOP GULP terminated with an error\n' in content:
                return False
        with open(pj(calcFolder, self.outputFile), 'rt') as fp:
            content = fp.readlines()
            for line in reversed(content):
                if ' Energy:' in line:
                    energy = line.split()[3]
                    try:
                        float(energy)
                    except ValueError:
                        pass
                    else:
                        return True

        try:
            with open(pj(calcFolder, self.optimizedStructure), 'rt') as fp:
                content = fp.readlines()
                for line in reversed(content):
                    if 'dump' in line:
                        return True
        except OSError:
            pass

        return False

    def readOutput(self, system, calcFolder : str):
        # TODO: implement http://qsh.ess.sunysb.edu:8000/trac/changeset/1255
        # Improve the GULP reader in case optimized_structure file is broken
        # Now ready to use parallel GULP  (applied to EX18-ZnOH)
        with open(pj(calcFolder, self.outputFile), 'rt') as f:
            content = f.readlines()

        if 'structure' in self.targetProperties:
            self.readStructure(system, content)
        if 'enthalpy' in self.targetProperties:
            system['enthalpy'] = self.readEnergy(content)
        if 'stressTensor' in self.targetProperties:
            system['stressTensor'] = self.readStressTensor(content)
        if 'strains' in self.targetProperties:
            system['strains'] = self.readStrains(content)
        if 'forces' in self.targetProperties:
            system['forces'] = self.readForces(content, len(system['molecules']))
        if 'elasticConstants' in self.targetProperties:
            system['elasticMatrix'] = self.readElasticMatrix(content)

    def readStructure(self, system, content):
        # This routine is to read crystal structure from GULP output
        # File: output
        # fractional for bulk
        # cartesian for surface

        assembledCell = system.pop('assembledCell')
        disassembler = system.pop('disassembler')

        # GULP prints the fractional coordinates before the Final lattice vectors
        # so they need to be stored and then atoms positions need to be set after we get the Final lattice vectors
        fractional_coordinates = None
        for i, line in enumerate(content):
            if line.find('Final cartesian coordinates of atoms') != -1:
                s = i + 5
                positions = []
                atomTypes = []
                while True:
                    s = s + 1
                    if content[s].find("------------") != -1:
                        break
                    if content[s].find(" s ") != -1:
                        continue
                    element, _, *xyz = content[s].split()[1:6]
                    XYZ = [float(x) for x in xyz]
                    positions.append(XYZ)
                    atomTypes.append(self.atomType(element))
                positions = np.array(positions)

            elif line.find('Final Cartesian lattice vectors') != -1:
                lattice_vectors = np.zeros((3, 3))
                s = i + 2
                for j in range(s, s + 3):
                    temp = content[j].split()
                    for k in range(3):
                        lattice_vectors[j - s][k] = float(temp[k])
                cell = self.cellType(lattice_vectors, pbc = assembledCell.getPBC())
                if fractional_coordinates is not None:
                    positions = cell.fractionalToCartesian(fractional_coordinates)

            elif line.find('Cartesian lattice vectors') != -1:
                lattice_vectors = np.zeros((3, 3))
                s = i + 2
                for j in range(s, s + 3):
                    temp = content[j].split()
                    for k in range(3):
                        lattice_vectors[j - s][k] = float(temp[k])
                cell = self.cellType(lattice_vectors, pbc = assembledCell.getPBC())
                if fractional_coordinates is not None:
                    positions = cell.fractionalToCartesian(fractional_coordinates)

            elif line.find('Final fractional coordinates of atoms') != -1:
                s = i + 5
                scaled_positions = []
                atomTypes = []
                while True:
                    s = s + 1
                    if content[s].find("------------") != -1:
                        break
                    if content[s].find(" s ") != -1:
                        continue
                    element, _, *xyz = content[s].split()[1:6]
                    XYZ = [float(x) for x in xyz]
                    scaled_positions.append(XYZ)
                    atomTypes.append(self.atomType(element))
                fractional_coordinates = np.asarray(scaled_positions)
                positions = assembledCell.fractionalToCartesian(fractional_coordinates)
        cell = cell.getEnvelopeCell(positions, 0)
        positions = cell.center(positions)
        structure = self.structureType(atomTypes, positions, cell = cell)
        system.update(disassembler.disassemble(structure))

    def readEnergy(self, content) -> float:
        energy_enthalpy = np.inf
        for line in content:
            m = re.match(r'\s*Total lattice en\S+\s*=\s*(-?[0-9.]+)\s*eV', line)
            if m:
                energy_enthalpy = float(m.group(1))
        if energy_enthalpy < np.inf:
            return energy_enthalpy
        else:
            raise RuntimeError('Read_GULP: GULP 1st SCF is not done, got bad value.')

    def readStressTensor(self, content):
        """
        NOTE 1: this value already with eternal pressure
        NOTE 2: value already taken to be a force on a cell
        :return stress tensor in GPa.
        """
        """
          Final stress tensor components (GPa):

        ---------------------------------------------------
        xx       -0.000001    yz        0.000010
        yy        0.000003    xz        0.000003
        zz        0.000011    xy       -0.000008
        ---------------------------------------------------
          Final stress tensor components (GPa):

        --------------------------------------------------------------------------------
        xx  **************    yz   345342.412237
        yy  -369708.222991    xz  -753092.817149
        zz  **************    xy   353195.477490
        --------------------------------------------------------------------------------

        """

        m = np.zeros([3, 3])
        for i, line in enumerate(content):
            if line.find('Final stress tensor components') != -1:
                m = [0., 0., 0., 0., 0., 0.]
                for j in range(3):
                    var = content[i + j + 3].split()[1]
                    m[j] = float(var)
                    var = content[i + j + 3].split()[3]
                    m[j + 3] = float(var)
                stress = np.array(m)
                break

        stress = np.array([[m[0], m[5], m[4]], [m[5], m[1], m[3]], [m[4], m[3], m[2]]])
        return stress

    def readStrains(self, content):
        strains = np.zeros(6)
        for i, line in enumerate(content):
            if line.find('Final strains') != -1:
                strains = [float(x) for x in content[i+2].split()]
                break
        return np.array(strains)

    def readForces(self, content, numAtoms : int):
        assert numAtoms > 0

        # force_orig = callAWK('GULP_force.awk', 'output', ['num=', num2str(numIons)]);
        readForces = False
        forces = []
        for i, line in enumerate(content):
            if readForces:
                index = i+5
                for i in range(numAtoms):
                    rawForces = content[index + i].split()[3:6]
                    forces.append([float(x) for x in rawForces])
                break

            if 'Final internal derivatives' in line:
                readForces = True

        return np.array(forces)

    def readElasticMatrix(self, content):
        elasticMatrix = np.zeros((6, 6), dtype=float)
        for i, line in enumerate(content):
            if 'Elastic Constant Matrix' in line:
                for row in content[i + 5: i + 11]:
                    elasticMatrix[i, :] = np.array(row.split()[1: 7], dtype=float)
                break
        else:
            raise RuntimeError("No elastic constant matrix information in the output.")
        return elasticMatrix

    def version(self):
        number = ''
        for line in open('gulp-old.output', 'r'):
            if line.lower().startswith('* Version'):
                number = line[12:17]
        return number

    @classmethod
    def registerTypes(cls, structureType, atomType, cellType, atomicDisassemblerType):
        cls.structureType = structureType
        cls.atomType = atomType
        cls.cellType = cellType
        cls.atomicDisassemblerType = atomicDisassemblerType