"""
USPEX.Stages.PWmat_Interface
============================

.. codeauthor:: Hao Li

"""
import logging
import sys
import shutil
import re
import numpy as np

from pathlib import Path
from .KPoints import KPoints, BadKPoints


logger = logging.getLogger(__name__)
EV_PER_CUBIC_ANGSTREM_PER_GPA = 1/160.21766208


class PWmat_Interface:
    '''
    Calculator for PWmat.
    Local running
    '''

    name = 'PWmat Calculator'
    shortname = 'pwmat'

    sleepTime = 30

    inputFile, outputFile, errorFile = 'input', 'output', 'error'

    # working files
    REPORT = 'REPORT'
    MOVEMENT = 'MOVEMENT'
    RELAXSTEPS = 'RELAXSTEPS'
    FINAL_CONFIG = 'final.config'

    structureType = None
    atomType = None
    cellType = None

    @classmethod
    def registerTypes(cls, structureType, atomType, cellType):
        cls.structureType = structureType
        cls.atomType = atomType
        cls.cellType = cellType

    def __init__(self, tag, etot_input, potcars, kresol, targetProperties: list = None, **kwargs):
        '''
        :param params: dictionary with parameters:
                * commandExecutable: str of executable command
                * kresol: float of K-points resolution
                * remote: dict of remote server params
                * taskManager: dict of task managers params
        :param step: int of current step
        '''


        self.tag = tag
        self.etot_input = Path(etot_input)
        assert self.etot_input.exists()

        self.potcars = [Path(potcar) for potcar in potcars]
        assert np.all([potcar.exists() for potcar in potcars])

        self.kPoints = KPoints(kresol)
        self.targetProperties = targetProperties if targetProperties is not None else ['structure', 'enthalpy']
        self.failedSystems = []


    def prepareLocalCalculation(self, system, calcFolder: Path):
        '''
        :param system: our system
        :return:
        '''
        structure = system['structure']

        cell = structure.getCell()
        system['pbc'] = cell.getPBC()

        atomTypes = structure.getAtomTypes()
        atomSymbols = [el.short_name for el in atomTypes]

        ############################# POTCAR ##################################
        try:
            # TODO varcomp ??? DO we need it here?
            # if self.state.varcomp or not os.path.exists('POTCAR_' + str(self.step)):  # we prefer this way
            f_potcar = (lambda pattern, filesname_list: [x for x in filesname_list if re.match(pattern, x)])
            for el in np.unique(atomSymbols):
                pattern = f'.*{el}.*UPF'
                potcarPath = f_potcar(pattern, self.potcars)[0]
                shutil.copy2(potcarPath, calcFolder)
        except:
            logging.debug('Insufficient POTCARs in Specific Directory')
            exc_info = sys.exc_info()
            raise exc_info[0].with_traceback(exc_info[1], exc_info[2])

        ############################## etot.input ##################################
        shutil.copy2(self.etot_input, calcFolder/'etot.input')

        # set mp_n123
        try:
            kPoints = self.kPoints.build(cell)
        except BadKPoints:
            # This LATTICE is extremely wrong, let's skip it from now
            logging.info('K poins cannot be built, so it\'s set as   [1, 1, 1]')
            kPoints = [1, 1, 1]

        # set IN.PSP
        with open(calcFolder/'etot.input', 'a') as fp:
            fp.write('\nMP_n123 = %d %d %d 0 0 0\n' % tuple(kPoints))

            files_in_calcFolder = [f.name for f in calcFolder.iterdir()]
            tmp_i = 1
            for el in np.unique(atomSymbols):
                pattern = f'{el}.*.UPF'
                PSP_name = f_potcar(pattern, files_in_calcFolder)
                INPSP = f'IN.PSP{tmp_i}={PSP_name[0]}\n'
                fp.write(INPSP)
                tmp_i += 1
        # set IN.RELAXOPT
        if system['externalPressure']:
            with open(calcFolder/'etot.input', 'a') as fp:
                fp.write('IN.RELAXOPT = T\n')
            with open(calcFolder/'IN.RELAXOPT', 'a') as fp:
                fp.write('PSTRESS_EXTERNAL= %10f\n' % (system['externalPressure']))



        ############################# atom.config ##################################

        # def writeAtomconfig(self,system : dict):
        with open(calcFolder/'atom.config', 'w') as fp:
            symbols = structure.getComposition()
            totalatom = 0
            for key, value in symbols.items():
                totalatom = totalatom + value

            fp.write(' %5d\n' % (totalatom))
            fp.write(' LATTICE\n')

            latt_form = ' %15.8f %14.8f %14.8f\n'
            cell_ = cell.getCellVectors()
            for i in range(3):
                latt = tuple(cell_[i])
                fp.write(latt_form % latt)

            fp.write(' POSITION\n')

            coord_form = '  %d %14.8f %14.8f %14.8f 1 1 1\n'
            coords = structure.getFractionalCoordinates()
            for i in range(totalatom):
                coord = (atomTypes[i].z,) + tuple(coords[i])
                fp.write(coord_form % coord)
            '''
            if system.externalPressure:
                fp.write('STRESS_EXTERNAL\n')
                fp.write('%g  %g  %g\n' % (system.externalPressure, 0., 0.))
                fp.write('%g  %g  %g\n' % (0., system.externalPressure, 0.))
                fp.write('%g  %g  %g\n' % (0., 0., system.externalPressure))
            '''
        return ''

############reading part
    def isConverged(self, calcFolder: Path):
        '''
        :param SYSTEM:
        :return: (bool) whether system calculation converged
        '''

        if not (calcFolder.joinpath(self.REPORT).exists() and
                calcFolder.joinpath(self.MOVEMENT).exists() and
                calcFolder.joinpath(self.RELAXSTEPS).exists()):
            return False
        # Checking whether converge

        with open(calcFolder/self.REPORT, 'r') as fp:
            content = fp.readlines()

        for line in content:
            if 'relax_detail' in line.lower():
                relax_detail= line
            if 'atom_move_step' in line.lower():
                atom_move_step_line = line

        #check the number of SCF steps:
        steps_specific = int(relax_detail.split()[3])
        atom_move_step = int(atom_move_step_line.split()[3])+1

        if atom_move_step < steps_specific:
            return True
        else:
            logging.error('PWmat SCF is not converged.')
            shutil.copy2(calcFolder/self.REPORT, calcFolder/f'ERROR{self.REPORT}')
            return False

    def readOutput(self, system, calcFolder: Path):
        """

        :rtype: object
        """
        if not calcFolder.joinpath('final.config').exists():
            shutil.copy(calcFolder/'atom.config', calcFolder/'final.config')
        with open(calcFolder/self.FINAL_CONFIG, 'r') as fp:
            content = fp.readlines()
        pbc = system.pop('pbc')
        atoms = int(content[0].split()[0])
        lat = []
        coor = []
        atomTypes = []
        for n, line in enumerate(content):
            if 'lattice' in line.lower():
                for i in range(3):
                    temp = content[n + 1 + i].split()
                    lat += [[float(temp[0]), float(temp[1]), float(temp[2])]]
            if 'position' in line.lower():
                for i in range(atoms):
                    temp = content[n + 1 + i].split()
                    atomTypes.append(self.atomType(int(temp[0])))
                    coor += [[float(temp[1]), float(temp[2]), float(temp[3])]]
        cell = self.cellType(lat, pbc)
        structure = self.structureType(atomTypes, coor, cell=cell)

        results = {}
        if 'structure' in self.targetProperties:
            results['structure'] = structure
        if 'enthalpy' in self.targetProperties:
            with open(calcFolder/self.REPORT, 'r') as fp:
                content = fp.readlines()
            results['enthalpy'] = self.readEnergy(content) + \
                                 cell.getVolume() * system['externalPressure'] * EV_PER_CUBIC_ANGSTREM_PER_GPA
        if 'stressTensor' in self.targetProperties:
            with open(calcFolder/self.MOVEMENT, 'r') as fp:
                content = fp.readlines()
            results['stressTensor'] = self.readPressureTensor(content)
        return results

    def readPressureTensor(self, content, index=-1):
        '''

        :param filename:
        :param index:
        :return:
        '''


        target = []
        for n, line in enumerate(content):
            if 'stress' in line and 'eV/natom'in line:
                stress = np.empty([3, 3], dtype=float)
                for i in range(3):
                    stress[i] = [float(num) for num in content[n + 1 + i].split()[4:]]
                target.append(stress)

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


    # TODO check this out
    def readEnergy(self, content):
        E_tot = 0
        # if all:
        #     E_tot = []

        for line in content:
        # Free energy
            if 'result' in line.lower() and 'e_tot' in line.lower():
                 # if all:
                 #     E_tot.append(float(line.split()[-1]))
                 # else:
                      E_tot = float(line.split()[-1])
        return E_tot

    def readForces(self, atoms, all=False):
        """Method that reads forces from MOVEMENT file.

        If 'all' is switched on, the forces for all ionic steps
        in the OUTCAR file be returned, in other case only the
        forces for the last ionic configuration is returned."""

        file = open('MOVEMENT', 'r')
        lines = file.readlines()
        file.close()
        n = 0
        if all:
            all_forces = []
        for n, line in enumerate(lines):
            if line.lower().startswith(' force'):
                forces = []
                for i in range(len(atoms)):
                    forces.append(np.array([float(force) for force in
                                            lines[n + 1 + i].split()[1:4]]))
                if all:
                    all_forces.append(np.array(forces)[self.resort])
            n += 1
        if all:
            return np.array(all_forces)
        else:
            return np.array(forces)[self.resort]

    def readFermi(self):
        energyFermi = None
        with open('REPORT', 'r') as f:
            for line in f.readlines():
                if 'E_Fermi' in line:
                    energyFermi = float(line.split()[1])
            return energyFermi
