"""
USPEX.Stages.GULP_Interface
===========================

"""

import logging
import numpy as np
import re
import shutil
import yaml

from pathlib import Path


logger = logging.getLogger(__name__)


class GULP_Interface:
    """
    Calculator for Gulp.
    Local running
    """

    DEFAULT_SLEEP_TIME = 10
    inputFile, outputFile, errorFile = 'input', 'output', 'error'

    def __init__(self, tag: str, ginput: str = None, goptions: str = None, libs: list[str] = None,
                 moleculeSpecifics: dict = None, fixCell: bool = False, targetProperties: list = None, **kwargs):
        """

        :param tag:
        :param ginput:
        :param goptions:
        :param libs:
        :param moleculeSpecifics:
        :param perturbate:
        :param fixCell:
        :param vacuumSize:
        :param targetProperties:
        :param kwargs:
        """

        self.tag = tag

        ginput = Path.cwd()/f'Specific/ginput_{tag}' if ginput is None else Path(ginput)
        assert ginput.exists()

        goptions = Path.cwd()/f'Specific/goptions_{tag}' if goptions is None else Path(goptions)
        assert goptions.exists()

        self.optimizedStructure = 'optimized.structure'

        with open(ginput, 'r') as f:
            self.ginput = f.read()
        with open(goptions, 'r') as f:
            self.goptions = f.read()

        self.libs = [Path(lib) for lib in libs] if libs else []
        assert all([lib.exists() for lib in self.libs])
        if moleculeSpecifics is not None:
            self.moleculeSpecifics = moleculeSpecifics
        else:
            self.moleculeSpecifics = {}

        self.fixCell = fixCell
        self.targetProperties = targetProperties

        logger.debug('GULP calculator created.')

    def prepareLocalCalculation(self, system, calcFolder: Path):
        """

        """

        structure = system.getProperty('structure', extension='atomistic')

        files_to_delete = [Path.cwd()/'output', Path.cwd()/'optimized.structure']
        for f in files_to_delete:
            if Path(f).is_file:
                Path(f).unlink(missing_ok=True)

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

        cell = structure.getCell()
        with open(calcFolder/'pbc', 'wt') as f:
            f.write(' '.join(f'{c}' for c in cell.getPBC()))
        lattice = type(cell)(cell.getCellVectors(), (1, 1, 1)).getCellParameters()

        content_to_write = ''
        content_to_write += 'cell\n'

        if self.fixCell:
            content_to_write += '%7.3f %7.3f %7.3f %7.3f %7.3f %7.3f 0 0 0 0 0 0 \n' % tuple(lattice)
        else:
            content_to_write += '%7.3f %7.3f %7.3f %7.3f %7.3f %7.3f\n' % tuple(lattice)

        content_to_write += 'fractional\n'

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

        for i, (symbol, coord) in enumerate(zip(structure.getAtomTypes(), structure.getFractionalCoordinates())):
            name = symbol.extra['gulpType'] if 'gulpType' in symbol.extra else symbol.short_name
            tuple_to_format = (name, ) +\
                              tuple(np.format_float_positional(c if not np.isclose(c, 0) else 0, unique=False,
                                                               precision=6) for c in coord)
            if cell.dim == 2:
                if i in system.getProperty('disassembler', extension='atomistic').allFixedIndices:
                    content_to_write += '%4s %12s %12s %12s 1 1 0 1 1 1\n' % tuple_to_format
                else:
                    content_to_write += '%4s %12s %12s %12s 1 1 0 0 0 0\n' % tuple_to_format
            elif symbol.charge is not None:
                tuple_to_format += (f'{symbol.charge:.3f}', )
                content_to_write += '%4s %12s %12s %12s core %8s\n' % tuple_to_format
            else:
                content_to_write += '%4s %12s %12s %12s\n' % tuple_to_format

        for i, j in structure.edges:
            content_to_write += f'connect    {i+1}   {j+1} \n'

        # Write part:
        total_content = self.goptions + '\n' + content_to_write + self.ginput + '\n'
        externalPressure = system.getProperty('externalPressure')
        if externalPressure >= 0.05:
            total_content += f"pressure {externalPressure:.1f}\n"
        total_content += 'dump every optimized.structure\n'

        with open(calcFolder/self.inputFile, 'wt') as f:
            f.write(total_content)

        with open(calcFolder/'extenededAtomTypes', 'wt') as f:
            f.write(''.join(f'- {element.extendedRepresentation()} \n' for element in structure.getAtomTypes()))

        for lib in self.libs:
            if lib.exists():
                shutil.copy(lib, calcFolder)

        logger.debug('GULP calculator prepared calculation.')

        return ''

    def isConverged(self, calcFolder: Path):
        """
        :param calcFolder:
        :return: whether optimization converged
        """
        with open(calcFolder/self.errorFile, 'rt') as fp:
            content = fp.readlines()
            if 'STOP GULP terminated with an error\n' in content:
                return False
        with open(calcFolder/self.outputFile, 'rt') as fp:
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
            with open(calcFolder/self.optimizedStructure, 'rt') as fp:
                content = fp.readlines()
                for line in reversed(content):
                    if 'dump' in line:
                        return True
        except OSError:
            pass

        return False

    def readOutput(self, system, calcFolder: Path):
        # TODO: implement http://qsh.ess.sunysb.edu:8000/trac/changeset/1255
        # Improve the GULP reader in case optimized_structure file is broken
        # Now ready to use parallel GULP  (applied to EX18-ZnOH)
        factory = system.getFactory()
        atomistic = factory.extensions['atomistic'].utility
        with open(calcFolder/self.outputFile, 'rt') as f:
            content = f.readlines()
        if (calcFolder / self.optimizedStructure).exists():
            with open(calcFolder / self.optimizedStructure, 'rt') as f:
                extra = f.readlines()
        else:
            extra = None
        with open(calcFolder/'extenededAtomTypes', 'rt') as f:
            extendedAtomTypes = yaml.safe_load(f.read())
        atomTypes = [atomistic.atomType(elementRep.pop('name'), **elementRep) for elementRep in extendedAtomTypes]

        result = factory()
        if 'structure' in self.targetProperties:
            with open(calcFolder/'pbc', 'rt') as f:
                pbc = tuple(int(c) for c in f.read().split())
            result.setProperty('structure', self.readStructure(atomTypes, atomistic, content, pbc, extra),
                               extension='atomistic')
        if 'enthalpy' in self.targetProperties:
            result.setProperty('enthalpy', self.readEnergy(content))
        if 'stressTensor' in self.targetProperties:
            result.setProperty('stressTensor', self.readStressTensor(content))
        if 'strains' in self.targetProperties:
            result.setProperty('strains', self.readStrains(content))
        if 'forces' in self.targetProperties:
            result.setProperty('forces', self.readForces(content, len(system['molecules'])))
        if 'dielectricTensor' in self.targetProperties:
            result.setProperty('dielectricTensor', self.readDielectricProperties(content))
        if 'elasticConstants' in self.targetProperties:
            result.setProperty('elasticMatrix', self.readElasticMatrix(content))
        return result

    @staticmethod
    def readStructure(atomTypes, atomistic, content, pbc, extra=None):
        # This routine is to read crystal structure from GULP output
        # File: output
        # fractional for bulk
        # cartesian for surface

        # GULP prints the fractional coordinates before the Final lattice vectors
        # so they need to be stored and then atoms positions need to be set after we get the Final lattice vectors
        fractional_coordinates = None
        for i, line in enumerate(content):
            if line.find('Final cartesian coordinates of atoms') != -1:
                s = i + 5
                positions = []
                # atomTypes = []
                while True:
                    s = s + 1
                    if content[s].find("------------") != -1:
                        break
                    if content[s].find(" s ") != -1:
                        continue
                    element, _, *xyz = content[s].split()[1:6]
                    XYZ = [float(x) for x in xyz]
                    positions.append(XYZ)
                    # atomTypes.append(atomistic.atomType(element))
                positions = np.array(positions)

            elif line.find('Final Cartesian lattice vectors') != -1:
                lattice_vectors = np.zeros((3, 3))
                s = i + 2
                for j in range(s, s + 3):
                    temp = content[j].split()
                    for k in range(3):
                        lattice_vectors[j - s][k] = float(temp[k])
                cell = atomistic.cellType(lattice_vectors, pbc=pbc)
                if fractional_coordinates is not None:
                    positions = cell.fractionalToCartesian(fractional_coordinates)

            elif line.find('Cartesian lattice vectors') != -1:
                lattice_vectors = np.zeros((3, 3))
                s = i + 2
                for j in range(s, s + 3):
                    temp = content[j].split()
                    for k in range(3):
                        lattice_vectors[j - s][k] = float(temp[k])
                cell = atomistic.cellType(lattice_vectors, pbc=pbc)
                if fractional_coordinates is not None:
                    positions = cell.fractionalToCartesian(fractional_coordinates)

            elif line.find('Final fractional coordinates of atoms') != -1:
                s = i + 5
                scaled_positions = []
                # atomTypes = []
                while True:
                    s = s + 1
                    if content[s].find("------------") != -1:
                        break
                    if content[s].find(" s ") != -1:
                        continue
                    element, _, *xyz = content[s].split()[1:6]
                    XYZ = [float(x) for x in xyz]
                    scaled_positions.append(XYZ)

                    # atomTypes.append(atomistic.atomType(element))
                fractional_coordinates = np.asarray(scaled_positions)
                positions = cell.fractionalToCartesian(fractional_coordinates)
        if extra is not None:
            bonds = []
            for line in extra:
                if 'connect' in line:
                    _, i, j, *_other = line.split()
                    bonds.append((int(i)-1, int(j)-1))
        else:
            bonds = None
        return atomistic.structureType(atomTypes, positions, cell=cell, edges=bonds)

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

    def readForces(self, content, numAtoms: int):
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

    def readDielectricProperties(self, content):
        Diel_Tens = np.zeros((3, 3))
        for i, line in enumerate(content):
            if 'Static dielectric constant' in line:
                for j, row in enumerate(content[i + 5, i + 8]):
                    Diel_Tens[j, :] = np.array(row.split()[1:4], dtype=float)
        return Diel_Tens

    def readElasticMatrix(self, content):
        elasticMatrix = np.zeros((6, 6), dtype=float)
        for i, line in enumerate(content):
            if 'Elastic Constant Matrix' in line:
                for j, row in enumerate(content[i + 5: i + 11]):
                    elasticMatrix[i, :] = np.array(row.split()[1: 7], dtype=float)
        return elasticMatrix

    def version(self):
        number = ''
        for line in open('gulp-old.output', 'r'):
            if line.lower().startswith('* Version'):
                number = line[12:17]
        return number
