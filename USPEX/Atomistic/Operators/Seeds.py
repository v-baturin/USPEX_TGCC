import logging
logger = logging.getLogger(__name__)

'''
@file        Seeds.py
@author:     Evgeny Tikhonov
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    e.tikhonov@physics.msu.ru
@date        15 October 2017
@brief       Pick Seeds VP Crystal structures.
'''


import numpy as np

from ase.io.vasp import read_vasp
from pathlib import Path
from time import time
from typing import List


class Seeds(object):

    systemRepresentationClass = None

    def __init__(self, utilities, generations:list=None, seedsFolders:list=None):
        '''

        :param config:
        :param generations:
        :param seedsFolders: default Seeds/POSCAR
        '''
        self.cellUtility = utilities.cellUtility
        self.compositionSpace = utilities.compositionSpace
        self.simpleMoleculeUtility = utilities.simpleMoleculeUtility
        self.bondUtility = utilities.bondUtility
        self.conditions = utilities.conditions
        self.atomistic = utilities.atomistic

        self.generations = generations if generations is not None else []
        self.seedsFolders = [Path(s) for s in seedsFolders] if seedsFolders is not None else []
        self.currentGeneration = 0

    def __call__(self, offspringFactory=None):

        if self.currentGeneration not in self.generations:
            logger.debug(f'No Seeds specified for generation {self.currentGeneration}.')
            self.currentGeneration += 1
            return ()

        ind = self.generations.index(self.currentGeneration)
        seedsFolder = self.seedsFolders[ind]

        if not seedsFolder.is_dir():
            logger.debug(f"Seeds folder {seedsFolder} doesn't exist.")
            self.currentGeneration += 1
            return ()

        seeds = []

        USUF = '.uspex' # Suffix of uspex files
        hasDesciption = np.any([USUF == filename.suffix for filename in seedsFolder.iterdir()])
        for filename in seedsFolder.iterdir():
            if (USUF == filename.suffix) == hasDesciption:
                if filename.is_file():
                    systems = self.atomistic.readAtomicStructures(filename)
                    for system in systems:
                        system = offspringFactory(**system)
                        structure = system.getProperty('structure', extension='atomistic')
                        minDistMatrix = self.bondUtility.getDistances(
                            structure.getAtomTypes(), self.conditions.externalPressure)
                        if self.simpleMoleculeUtility.checkMinDistances(system, minDistMatrix):
                            self.conditions.putConditions(system)
                            system.setProperty('filename', filename)
                            seeds.append(system)
                        else:
                            logger.info(f"Structure created from seed {filename} violates constraints.")

        self.currentGeneration += 1
        return tuple(seeds)

