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


class Autofrac(object):
    '''

    '''

    def __init__(self, fractions : Dict[str, tuple], population : list, best : list, newFoundSystems : list, varOperators : list):
        '''

        :param population:
        :param best:
        :param newFoundSystems:
        :param varOperators:
        '''

        self.weightsLast = Counter()
        for system in population:
            self.weightsLast[system['howCome']] += 1
        self.weightsBest = Counter()
        for system in best:
            if system not in newFoundSystems:
                self.weightsBest[system['howCome']] += 1

        self.initFracs = {}
        self.minFracs = {}
        self.maxFracs = {}
        for VO in varOperators:
            name = VO.__class__.__name__[0].lower() + VO.__class__.__name__[1:]
            if name in fractions:
                self.minFracs[VO], self.maxFracs[VO], self.initFracs[VO] = fractions[name]
            else:
                self.minFracs[VO], self.maxFracs[VO], self.initFracs[VO] = 0.0,0.0,0.0

    def howMany(self, varOperator, leftPopSize : int):
        '''

        :param varOperator:
        :param leftPopSize:
        :return:
        '''

        if self.weightsLast[varOperator] == 0:
            initialNorm = sum(self.initFracs.values())
            frac = self.initFracs[varOperator] / initialNorm if initialNorm > 0 else 0
            howMany = np.floor(frac * leftPopSize)
        else:
            minimalNorm = sum(self.minFracs.values())
            lastNorm = np.fromiter((value for value in self.weightsLast.values()), dtype=int).sum()
            # bestNorm = np.fromiter((value for value in self.weightsBest.values()), dtype=int).sum()
            weightsNorm = np.fromiter(
                (bestN ** 2 / lastN for lastN, bestN in zip(self.weightsLast.values(), self.weightsBest.values())),
                dtype=float).sum()
            weight = self.weightsBest[varOperator] ** 2 / self.weightsLast[varOperator]
            frac = max(self.minFracs[varOperator] / minimalNorm,
                       (self.weightsLast[varOperator] / lastNorm + weight / weightsNorm) / 2)
            howMany = np.floor(frac * leftPopSize)

        del self.weightsLast[varOperator]
        del self.weightsBest[varOperator]
        del self.initFracs[varOperator]
        del self.minFracs[varOperator]
        del self.maxFracs[varOperator]
        return int(howMany)
