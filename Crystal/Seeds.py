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

from ..VarOperator import VarOperator, VOFailed


class Seeds(VarOperator):

    def __init__(self, systemFactory, config, pool, utilities, generations:list=None, seedsFolders:list=None):
        '''

        :param config:
        :param generations:
        :param seedsFolders: default Seeds/POSCAR
        '''
        super(Seeds, self).__init__(systemFactory, config, pool, utilities)
        self.generations = generations if generations is not None else []
        self.seedsFolders = seedsFolders if seedsFolders is not None else []
        self.active = False
        self.currentGeneration = 0

    def prepare(self):
        if self.active:
            logger.debug('Warning: Seeds was not deactivated properly.')
        self.active = True

    def standby(self):
        if self.active:
            logger.debug(f'Warning: Seeds was not called in generation {self.currentGeneration}.')
            self.active = False
        self.currentGeneration += 1

    def __call__(self):
        if not self.active:
            logger.debug('Warning: Seeds was already deactivated for this generation')
            raise VOFailed

        self.active = False

        if self.currentGeneration not in self.generations:
            logger.debug(f'No Seeds specified for generation {self.currentGeneration}.')
            raise VOFailed

        ind = self.generations.index(self.currentGeneration)
        seedsFolder = self.seedsFolders[ind]

        if not os.path.isdir(seedsFolder):
            raise VOFailed

        seeds = []

        for filename in os.listdir(seedsFolder):
            filename = os.path.join(seedsFolder, filename)
            if os.path.isfile(filename):
                with open(filename, "rt") as f:
                    try:
                        system_dict = toml.load(f)
                    except:
                        continue
                system = self.systemFactory.fromDICT(system_dict, old=False)
                system.config = self.config
                if system.isGoodSystem():
                    seeds.append(system)
                    self.pool.assignID(system)
                    system.howCome = self.name
                    system.parent = 'None'
                    logger.info(f"Structure {system.ID} created from seed {filename}.")
                else:
                    logger.info(f"Structure created from seed {filename} violates constraints.")

        return tuple(seeds)
