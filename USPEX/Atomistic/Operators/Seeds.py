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
import os

from ase.io.vasp import read_vasp
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
        self.environmentUtility = utilities.environmentUtility
        self.ionDistances = utilities.ionDistances
        self.conditions = utilities.conditions

        self.generations = generations if generations is not None else []
        self.seedsFolders = seedsFolders if seedsFolders is not None else []
        self.currentGeneration = 0

    def __call__(self):

        if self.currentGeneration not in self.generations:
            logger.debug(f'No Seeds specified for generation {self.currentGeneration}.')
            self.currentGeneration += 1
            return ()

        ind = self.generations.index(self.currentGeneration)
        seedsFolder = self.seedsFolders[ind]

        if not os.path.isdir(seedsFolder):
            logger.debug(f"Seeds folder {seedsFolder} doesn't exist.")
            self.currentGeneration += 1
            return ()

        seeds = []

        filenames = os.listdir(seedsFolder)
        hasDesciption = np.any(['.uspex' in filename for filename in filenames])
        for filename in filenames:
            if ('.uspex' in filename) == hasDesciption:
                filename = os.path.join(seedsFolder, filename)
                if os.path.isfile(filename):
                    systems = self.systemRepresentationClass.readAtomicStructures(filename, self.environmentUtility)
                    for system in systems:
                        atomSymbols, atomDistances, disassembler = self.simpleMoleculeUtility.getMinDistances(**system)
                        minDistMatrix = self.ionDistances.getDistances(atomSymbols, self.conditions.externalPressure)
                        if disassembler.environment is not None:
                            inds = disassembler.envIndices
                            atomDistances[tuple(np.meshgrid(inds, inds))] = minDistMatrix[tuple(np.meshgrid(inds, inds))]
                        if np.all(atomDistances >= minDistMatrix):
                            self.conditions.putConditions(system)
                            system['filename'] = filename
                            seeds.append(system)
                        else:
                            logger.info(f"Structure created from seed {filename} violates constraints.")

        self.currentGeneration += 1
        return tuple(seeds)

    @classmethod
    def registerTypes(cls, systemRepresentationClass):
        cls.systemRepresentationClass = systemRepresentationClass