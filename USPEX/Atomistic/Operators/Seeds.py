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
import toml

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

        for filename in os.listdir(seedsFolder):
            filename = os.path.join(seedsFolder, filename)
            if os.path.isfile(filename):
                with open(filename, "rt") as f:
                    system = self.systemRepresentationClass.readAtomicStructure(f, pbc=self.cellUtility.getPBC())
                molecules = system['molecules']
                cell = system['cell']
                atomSymbols, atomDistances = self.simpleMoleculeUtility.getMinDistances(molecules, cell)
                minDistMatrix = self.ionDistances.getDistances(atomSymbols, self.conditions.externalPressure)
                composition = self.simpleMoleculeUtility.composition(system)
                if np.all(atomDistances >= minDistMatrix) and self.compositionSpace.isGoodComposition(composition):
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