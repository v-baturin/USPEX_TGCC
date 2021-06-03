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

    def __init__(self, systemFactory, config, pool, utilities, generations:list=None, seedsFolders:list=None):
        '''

        :param config:
        :param generations:
        :param seedsFolders: default Seeds/POSCAR
        '''
        self.systemFactory = systemFactory
        self.config = config
        self.pool = pool
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
                try:
                    with open(filename, "rt") as f:
                        system_dict = toml.load(f)
                    system = self.systemFactory.fromDICT(system_dict, old=False)
                    system.config = self.config
                except:
                    try:
                        tmp = read_vasp(filename)
                        system = self.systemFactory(symbols = tmp.get_chemical_symbols(),
                                                    cell = tmp.get_cell(),
                                                    positions = tmp.get_positions(), **self.config)
                    except:
                        continue
                if system.isGoodSystem():
                    system = {'structure': system}
                    seeds.append(system)
                    self.pool.assignID(system)
                    system['howCome'] = self.__class__.__name__
                    system['parent'] = 'None'
                    logger.info(f"Structure {system['ID']} created from seed {filename}.")
                else:
                    logger.info(f"Structure created from seed {filename} violates constraints.")

        self.currentGeneration += 1
        return tuple(seeds)
