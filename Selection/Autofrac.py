'''
@file        Autofrac.py
@author:     Pavel Bushlanov
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    paulbush@mail.ru
@date        December 2016
@brief       Class for all engines in USPEX code
'''

'''
FunctionFolder/USPEX/src/update_STUFF.m
'''

import numpy as np
from collections import Counter
from typing import Dict
from copy import copy


class Autofrac(object):
    '''

    '''

    def __init__(self, fractions : Dict[str, tuple], weightsLast, best : list, newFoundSystems : list, varOperators : list):
        '''

        :param population:
        :param best:
        :param newFoundSystems:
        :param varOperators:
        '''

        self.weightsLast = copy(weightsLast)
        self.weightsBest = Counter()
        for system in best:
            if system['howCome'] != 'Seeds' and system in newFoundSystems:
                self.weightsBest[system['howCome']] += 1

        self.initWeights = {}
        self.minFracs = {}
        self.maxFracs = {}
        for VO in varOperators:
            name = type(VO).__name__
            nl = name[0].lower() + name[1:]
            self.minFracs[name], self.maxFracs[name], self.initWeights[name] = fractions[nl] if nl in fractions else (0.0, 0.0, 0.0)

    def howMany(self, howCome, leftPopSize : int, totalPopSize : int):
        '''

        :param howCome:
        :param leftPopSize:
        :return:
        '''

        if self.weightsLast[howCome] == 0:
            initialNorm = sum(self.initWeights.values())
            frac = self.initWeights[howCome] / initialNorm if initialNorm > 0 else 0
            howMany = np.floor(frac * leftPopSize)
        else:
            lastNorm = np.fromiter((value for value in self.weightsLast.values()), dtype=int).sum()
            # bestNorm = np.fromiter((value for value in self.weightsBest.values()), dtype=int).sum()
            weightsNorm = np.fromiter(
                (bestN ** 2 / lastN for lastN, bestN in zip(self.weightsLast.values(), self.weightsBest.values())),
                dtype=float).sum()
            weight = self.weightsBest[howCome] ** 2 / self.weightsLast[howCome]
            frac = (self.weightsLast[howCome] / lastNorm + weight / weightsNorm) / 2
            howMany = np.floor(frac * leftPopSize)
            howManyMin = np.floor(self.minFracs[howCome] * totalPopSize)
            howManyMax = np.floor(self.maxFracs[howCome] * totalPopSize)
            howMany = max(howManyMin, howMany)
            howMany = min(howManyMax, howMany)

        del self.weightsLast[howCome]
        del self.weightsBest[howCome]
        del self.initWeights[howCome]
        del self.minFracs[howCome]
        del self.maxFracs[howCome]
        return int(howMany)
