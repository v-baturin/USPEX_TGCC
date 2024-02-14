"""
USPEX.Stages.VASP_Interface
===========================

"""

import os
import logging
import numpy as np
import shutil
import re
import yaml

from pathlib import Path
from typing import List
from ase.io.vasp import iread_vasp_out, read_vasp_xml, write_vasp
from ase.io import ParseError
from ase.atoms import Atoms
from ase.constraints import FixAtoms


from .KPoints import KPoints, BadKPoints

logger = logging.getLogger(__name__)

SUPERCELL_MIN_SIZE = 10.
KJMOL_IN_EV = 0.01036410

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
    phonopy_specific = 'phonopy_files'

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
        
        ############################# SUPERCELL_SHAPE #########################
        supercell_dim = self.getSupercellShape(structure)
        
        ############################# KPOINTS #################################

        try:
            kPoints = np.ceil(np.array(self.kPoints.build(structure.getCell())) / supercell_dim)
        except BadKPoints:
            # This LATTICE is extremely wrong, let's skip it from now
            logger.info('K-points cannot be built, so it\'s set as   [1, 1, 1]')
            kPoints = [1, 1, 1]

        with open(calcFolder/self.kpoints_file, 'w') as fp:
            fp.write('EA\n0\nGamma\n')
            fp.write('%4d %4d %4d\n' % tuple(kPoints))

        ############################# COPY PHONOPY FILES #######################
        shutil.copytree(self.phonopySupplemPath, calcFolder, dirs_exist_ok=True)
        
        ############################# MODIFY_PHONOPY_SCRIPT ####################
        supercell_str = " ".join([str(i) for i in supercell_dim])

        with open(self.phRunscriptTemplatePath, 'r') as file:
            filedata = file.readlines()
            for i, line in enumerate(filedata):
                if '%DIM' in line and 'phonopy' in line:
                    filedata[i] = f'phonopy -d --dim="{supercell_str}"'

        # Write the file out again
        with open(calcFolder / self.phRunscriptTemplatePath.name, 'w') as file:
            file.writelines(filedata)
        os.chmod(calcFolder / self.phRunscriptTemplatePath.name, 0o777)

        ############################# WRITE_CONF_FILES ######################
        with open(calcFolder / self.poscar_file, 'r') as poscar_fid:
            atom_name = ' '.join(poscar_fid.readlines()[5].split())

        k_path_coords, labels = self.getKStrings(structure)

        with open(calcFolder / self.meshConf, 'w') as meshconf, open(calcFolder / self.bandConf, 'w') as bandconf:
            meshLines = [f'ATOM_NAME = {atom_name}',
                         f'DIM = {supercell_str}',
                         f'MP = 40 40 40']
            bandLines = [f'ATOM_NAME = {atom_name}',
                         f'DIM = {supercell_str}',
                         f'BAND = {k_path_coords}',
                         f'BAND_LABELS = {labels}']
            meshconf.write('\n'.join(meshLines))
            bandconf.write('\n'.join(bandLines))
        
        return ''

    def __init__(self, tag: str,
                 kresol: float,
                 incar: str | os.PathLike = None,
                 potcarsPath: str | os.PathLike = None,
                 phRunscriptTemplatePath:  str | os.PathLike = None,
                 targetProperties: list = None,
                 supercellMinSize: float = None,
                 bandConf: str | os.PathLike = None,
                 meshConf: str | os.PathLike = None,
                 phonopyFiles: str| os.PathLike = None,
                 **kwargs):

        self.tag = tag
        self.incar = Path(incar) if incar is not None else Path.cwd()/f'Specific/INCAR_{tag}'
        assert self.incar.exists()

        self.potcarsPath = Path(potcarsPath) if potcarsPath is not None else Path.cwd()/'Specific'
        assert self.potcarsPath.exists()

        self.phonopySupplemPath = Path(phonopyFiles) if phonopyFiles else Path.cwd() / 'Specific/phonopy_files'
        assert self.phonopySupplemPath.exists()
        
        self.phRunscriptTemplatePath =(
            Path(phRunscriptTemplatePath)) if phRunscriptTemplatePath else self.phonopySupplemPath / 'script_phonopy.sh'
        assert self.phRunscriptTemplatePath.exists()

        self.bandConf = bandConf if bandConf else 'band.conf'
        self.meshConf = meshConf if meshConf else 'mesh.conf'
        
        

        self.kPoints = KPoints(kresol)
        self.failedSystems = []

        self.supercellMinSize = supercellMinSize
        self.targetProperties = targetProperties

    def isConverged(self, calcFolder: Path):
        '''
        :param calcFolder:
        :return: (bool) whether system calculation converged
        '''
        return Path.exists(calcFolder / 'thermal_properties.yaml')

    ############reading part

    def readOutput(self, system, calcFolder: Path):
        with open(calcFolder / 'thermal_properties.yaml', 'r') as f:
            ph_results = yaml.safe_load(f.read())
        factory = system.getFactory()
        result = factory()
        for property in self.targetProperties:
            if property.casefold() in ('zpe', 'zero_point_energy'):
                result.setProperty('ZPE', ph_results['zero_point_energy'] * KJMOL_IN_EV)
            else:
                logger.warning('Only ZPE is available in phonopy calculation')
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

    def getSupercellShape(self, structure):
        if self.supercellMinSize is None:
            supercell_min_size = SUPERCELL_MIN_SIZE
        else:
            supercell_min_size = self.supercellMinSize
        atoms, _ = self.structure2Atoms(structure)
        cell = atoms.cell
        supercell_shape = []
        vol = np.abs(np.linalg.det(cell))
        for i in range(3):
            j, k = np.roll(np.arange(3), -i)[1:]
            area = np.linalg.norm(np.cross(cell[j], cell[k]))
            supercell_shape.append(int(np.ceil(supercell_min_size / (vol / area))))

        return np.array(supercell_shape)

    def getKStrings(self, structure):
        atoms, _ = self.structure2Atoms(structure)
        lat = atoms.cell.get_bravais_lattice()
        special_path = lat.special_path.split(',')[0]
        pattern = re.compile(r'([A-Z]\d*)')
        points_sequence = pattern.findall(special_path)
        specialKPathCoords = [lat.get_special_points()[label] for label in points_sequence]
        kCoordsStr = '  '.join([' '.join([f'{ki:1.3f}' for ki in k]) for k in specialKPathCoords])
        labelsString = ' '.join(points_sequence).replace('G', '$\Gamma$')
        return kCoordsStr, labelsString


