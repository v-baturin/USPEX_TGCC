'''
@file        QE_Calculator.py
@author:     Artem Samtsevich
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    samtsevichartem@gmail.com
@date        30 August 2016
@brief       Class for calculator of Quantum Espresso
'''
import logging
logger = logging.getLogger(__name__)

import os
import shutil
import numpy as np
from os.path import join as pj
from ase.io.espresso import read_espresso_out

from .Common.KPoints import KPoints, BadKPoints
from .Common.SHELL_Interface import SHELL_Interface


class QE_Interface(SHELL_Interface):
    '''
    Calculator for QE.
    Local running
    '''


    SPECIFIC_FOLDER = os.getcwd() + '/Specific'

    _DEFAULT_SLEEP_TIME = 30

    def __init__(self, tag : str, kresol : float, options: str = None, libs: list = None, **kwargs):
        '''

        :param tag: tag of the stage
        :param kresol: float of K-points resolution
        :param options:(str) path to qEspresso_options-file.
        :param libs: (list) list of paths to interatomic potentials.
        :param kwargs:
        '''
        super().__init__(**kwargs)

        if options is not None:
            self.options = options
        else:
            self.options = pj(os.getcwd(), f'Specific/qEspresso_options_{tag}')
        self.libs = libs if libs else []
        self.kPoints = KPoints(kresol)

    def readOutput(self, system : dict, calcFolder: str):
        disassembler = system['disassembler']
        del system['disassembler']
        structure = system['structure']
        del system['structure']

        with open(pj(calcFolder, self.outputFile), 'rt') as f:
            tmp = next(read_espresso_out(f, index=slice(None, -2, -1)))
        if tmp:
            cell = structure.getCell()
            system.update(disassembler.disassemble(type(structure)(structure.getAtomTypes(), tmp.get_positions(),
                                                                   cell = type(cell)(tmp.get_cell().array, cell.getPBC()))))

            system['enthalpy'] = tmp.get_calculator().results['energy']
            # system.forces = np.copy(tmp.get_calculator().results['forces'])

    def prepareLocalCalculation(self, system: dict, calcFolder: str):
        molecules = system['molecules']
        cell = system['cell']
        systemFactory = type(molecules[0])
        structure, disassembler = systemFactory.assemble(molecules, cell = cell)
        system['structure'] = structure
        system['disassembler'] = disassembler


        atomTypes = structure.getAtomTypes()
        numIons_size = len(np.unique(atomTypes))

        with open(self.options, 'rt') as source:
            data = source.readlines()
        for i, line in enumerate(data):
            if 'AAAA' in line:
                data[i] = line.replace('AAAA', '{}'.format(len(structure)))
            elif 'BBBB' in line:
                data[i] = line.replace('BBBB', '{}'.format(numIons_size))


        data.append('CELL_PARAMETERS cubic\n')

        BOHR = 0.52917721067  # Angstrom
        #lat = latConverter(latConverter(LATTICE)) / BOHR
        lat = structure.getCell().getCellVectors() / BOHR#_lengths_and_angles()

        data.append('{:8.4f} {:8.4f} {:8.4f}\n'.format(*lat[0, :]))
        data.append('{:8.4f} {:8.4f} {:8.4f}\n'.format(*lat[1, :]))
        data.append('{:8.4f} {:8.4f} {:8.4f}\n'.format(*lat[2, :]))

        data.append('ATOMIC_POSITIONS {crystal} \n')

        for symbol, coord in zip(atomTypes, structure.getFractionalCoordinates()):
            data.append('{:4s} {:12.6f} {:12.6f} {:12.6f}\n'.format(symbol.short_name, *coord))


        ############################# KPOINTS #################################
        try:
            kPoints = self.kPoints.build(structure)
        except BadKPoints:
            # This LATTICE is extremely wrong, let's skip it from now
            logger.info('K-points cannot be built, so it\'s set as   [1, 1, 1]')
            kPoints = [1,1,1]

        data.append('K_POINTS {automatic} \n')
        data.append('{:4d} {:4d} {:4d}  0 0 0\n'.format(*kPoints))


        with open(pj(calcFolder, self.inputFile), 'wt') as dest:
            dest.write('\n'.join(data))

        for lib in self.libs:
            if isinstance(lib,str) and os.path.exists(lib):
                shutil.copy(lib, calcFolder)


    def isConverged(self, calcFolder: str):
        if not os.path.exists(pj(calcFolder, self.outputFile)):
            res = False
        else:
            with open(pj(calcFolder, self.outputFile), 'rt') as out:
                res = 'JOB DONE' in out.read()
        if not res:
            logger.error('Quantum Espresso is not completely Done')
        return res
