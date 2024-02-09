"""
USPEX.Stages.VASP_Interface
===========================

"""

import logging
import numpy as np
import shutil

from pathlib import Path
from typing import List
from ase.io.vasp import iread_vasp_out, read_vasp_xml, write_vasp
from ase.io import ParseError
from ase.atoms import Atoms
from ase.constraints import FixAtoms


from .KPoints import KPoints, BadKPoints

logger = logging.getLogger(__name__)

SUPERCELL_MIN_SIZE = 10.

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


class PHONOPY_Interface:
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


    AtomicStructureRepresentation = None

    @classmethod
    def registerTypes(cls, AtomicStructureRepresentation):
        cls.AtomicStructureRepresentation = AtomicStructureRepresentation

    def __init__(self, tag: str,
                       kresol: float,
                       incar: str = None,
                       potcarsPath: str = None,
                       targetProperties: list = None,
                       supercellMinSize: float = None,
                       **kwargs):

        self.tag = tag
        self.incar = Path(incar) if incar is not None else Path.cwd()/f'Specific/INCAR_{tag}'
        assert self.incar.exists()

        self.potcarsPath = Path(potcarsPath) if potcarsPath is not None else Path.cwd()/'Specific'
        assert self.potcarsPath.exists()

        self.phonopyTemplatesPath = Path.cwd()/f'Specific/phonopy_templates/'

        self.kPoints = KPoints(kresol)
        self.failedSystems = []

        self.supercellMinSize = supercellMinSize
        self.targetProperties = targetProperties

    def prepareLocalCalculation(self, system, calcFolder: Path):
        '''
        :param system: our system
        :param calcFolder: calculation folder
        :return:
        '''
        with open(calcFolder/self.inputFile, 'wt') as f:
            pass

        structure = system.getProperty('structure', extension='atomistic')
        cell = structure.getCell()
        with open(calcFolder/'pbc', 'wt') as f:
            f.write(' '.join(f'{c}' for c in cell.getPBC()))


        ############################# POSCAR ##################################

        disassembler = system.getProperty('disassembler', extension='atomistic')
        self.write(structure, disassembler.allFixedIndices, f"EA{system['.ID']}", calcFolder)

        ############################## INCAR ##################################
        shutil.copy2(self.incar, calcFolder/self.incar_file)

        externalPressure = system.getProperty('externalPressure')
        if externalPressure:
            with open(calcFolder/self.incar_file, 'a') as myfile:
                myfile.write(f"\nPSTRESS={10 * externalPressure:10f}\n")
        if calcFolder in self.failedSystems:
            with open(calcFolder/self.incar_file, 'r') as myfile:
                incar_data = myfile.readlines()
            for i, incar_line in enumerate(incar_data):
                if 'symprec' in incar_line.casefold() and incar_line.strip()[0] != '#':
                    current_symprec = float(incar_line.split('=')[1].strip())
                    new_symprec = 1.5 * current_symprec
                    incar_data[i] = f'SYMPREC   =  {new_symprec:1.1E}\n'
                    break
            else:
                incar_data.append(f'SYMPREC   =  1E-4\n')
            with open(calcFolder/self.incar_file, 'w') as myfile:
                myfile.writelines(incar_data)

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
            fp.write('EA\n0\nGamma\n')
            fp.write('%4d %4d %4d\n' % tuple(kPoints))


        ############################# MODIFY_PHONOPY_SCRIPTS ####################
        supercell_dim = self.getSupercellForPhonopy(structure)

        with open(self.phonopyTemplatesPath / 'phonopy_script.sh', 'r') as file:
            filedata = file.readlines()
            for i, line in enumerate(filedata):
                if '%DIM' in line and 'phonopy' in line:
                    filedata[i] = f'phonopy -d --dim="{" ".join(str(i) for i in supercell_dim)}"'

        # Write the file out again
        with open('file.txt', 'w') as file:
            file.writelines(filedata)

        ############################# MODIFY_CONF_TEMPLATES ######################

        supercell_dim = self.getSupercellForPhonopy(structure)



        return ''

    def isConverged(self, calcFolder: Path):
        '''
        :param calcFolder:
        :return: (bool) whether system calculation converged
        '''

        if not (calcFolder.joinpath(self.outcar_file).exists() and
                calcFolder.joinpath(self.oszicar_file).exists() and
                calcFolder.joinpath(self.contcar_file).exists()):
            return False

        # Checking the real vs reciprocal lattice inconsistency error


        # Checking whether converge
        NELM = -1
        row_number = None
        with open(calcFolder/self.oszicar_file, 'r') as f:
            content = f.readlines()
            for i in range(len(content)):
                if content[i].find(' F= ') >= 0:
                    row_number = i
            if row_number is None:
                with open(calcFolder / self.outcar_file, 'r') as outcar_fid:
                    for line in outcar_fid:
                        if 'Inconsistent Bravais lattice types found for crystalline and' in line:
                            logger.error('VASP SCF is not converged.')
                            shutil.copy2(calcFolder / self.outcar_file, calcFolder / f'ERROR-{self.outcar_file}')
                            self.failedSystems.append(calcFolder)


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
        with open(calcFolder / 'pbc', 'rt') as f:
            pbc = tuple(int(c) for c in f.read().split())
        trajectory = self.read(calcFolder, pbc)
        results = trajectory[-1]['results']
        factory = system.getFactory()
        result = factory()

        if 'structure' in self.targetProperties:
            result.setProperty('structure', trajectory[-1]['structure'], extension='atomistic')
        if 'enthalpy' in self.targetProperties:
            if 'enthalpy' in results:
                result.setProperty('enthalpy', results['energy'])
            else:
                result.setProperty('energy', results['energy'])
        if 'energy' in self.targetProperties:
            result.setProperty('energy', results['energy'])
        if 'forces' in self.targetProperties:
            result.setProperty('forces', results['forces'])
        if 'trajectory' in self.targetProperties:
            # for subsystem in trajectory:
            #     subsystem['disassembler'] = system['disassembler']
            #     subsystem['externalPressure'] = system['externalPressure']
            result.setProperty('trajectory', trajectory)

        with open(calcFolder/self.outcar_file, 'rt') as fp:
            content = fp.readlines()
        if 'stressTensor' in self.targetProperties:
            result.setProperty('stressTensor', self.readPressureTensor(content))
        if 'dielectricTensor' in self.targetProperties:
            result.setProperty('dielectricTensor', self.readDielectricProperties(content))
        if 'dipoleMoment' in self.targetProperties:
            result.setProperty('dipoleMoment', self.readDipoleMoment(content))
        if 'energyFermi' in self.targetProperties:
            result.setProperty('energyFermi', self.readFermi(content))
        if 'elasticConstants' in self.targetProperties:
            result.setProperty('elasticConstants', self.readElasticMatrix(content))
        return result

    def structure2Atoms(self, structure):
        cell = structure.getCell()
        symbols = np.asarray([el.short_name for el in structure.getAtomTypes()])
        order = np.argsort(symbols)
        return Atoms(symbols[order], structure.getCartesianCoordinates()[order], cell=cell.getCellVectors()), order

    def write(self, structure, fixedIndices, label, calcFolder: Path):
        atoms, order = self.structure2Atoms(structure)
        if len(fixedIndices) > 0:
            atoms.set_constraint(FixAtoms(indices=np.argsort(order)[fixedIndices]))
        write_vasp(calcFolder/self.poscar_file, atoms, label=label, direct=True, vasp5=True, long_format=True)
        with open(calcFolder/'symbolsOrder', 'wt') as f:
            f.write(' '.join(f'{c}' for c in order))

    def read(self, calcFolder: Path, pbc):
        with open(calcFolder/'symbolsOrder', 'rt') as f:
            symbolsOrder = tuple(int(c) for c in f.read().split())
        try:
            with open(calcFolder/self.outcar_file) as f:
                trajectoryAtoms = list(iread_vasp_out(f, None))
        except (KeyError, ParseError):
            with open(calcFolder/self.xml_file) as f:
                trajectoryAtoms = list(read_vasp_xml(f))
        trajectory = []
        for atoms in trajectoryAtoms:
            size = len(atoms)
            positions = np.empty((size, 3), dtype=float)
            atomTypes = np.empty(size, dtype=self.AtomicStructureRepresentation.atomType)
            for i, symbol, position in zip(symbolsOrder, atoms.get_chemical_symbols(), atoms.get_positions()):
                positions[i] = position
                atomTypes[i] = self.AtomicStructureRepresentation.atomType(symbol)
            cell = self.AtomicStructureRepresentation.cellType(atoms.get_cell().array, pbc)
            structure = self.AtomicStructureRepresentation.structureType(atomTypes, positions, cell=cell)

            trajectory.append(dict(
                structure=structure,
                results=atoms.get_calculator().results
            ))
        return trajectory

    def getSupercellForPhonopy(self, structure):
        if self.supercellMinSize is None:
            supercell_min_size = SUPERCELL_MIN_SIZE
        else:
            supercell_min_size = self.supercellMinSize
        atoms, _ = self.structure2Atoms(structure)
        cell = atoms.cell
        supercell_shape = []
        vol = np.abs(np.linalg.det(cell))
        for i in range(3):
            j, k = np.roll(np.arange(2), i)[0:2]
            area = np.linalg.norm(np.cross(cell[j], cell[k]))
            supercell_shape.append(int(np.ceil(supercell_min_size / (vol / area))))

        return supercell_shape

