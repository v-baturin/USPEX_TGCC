'''
@file        VarOperator.py
@author:     Pavel Bushlanov
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    paulbush@mail.ru
@date        21 July 2016
@brief       Abstract class for different variation operators
'''


from abc import ABCMeta, abstractmethod

from USPEX.Common.Config import Config


class VarOperator(object):
    '''

    '''

    __metaclass__ = ABCMeta

    name = None
    isActive = True

    def __init__(self, config : Config, initFrac : float, minFrac : float=0.1, maxFrac : float=1.0):
        '''

        :param config:
        :param initFrac:
        :param minFrac:
        :param maxFrac:
        '''

        self.config = config
        assert minFrac >= 0.0 and minFrac <= 1.0
        assert maxFrac >= 0.0 and maxFrac <= 1.0
        assert minFrac <= maxFrac
        self._minimalFraction = minFrac
        self._maximalFraction = maxFrac
        self._initialFraction = initFrac

    @property
    def initialFraction(self):
        return self._initialFraction

    @property
    def minimalFraction(self):
        return self._minimalFraction

    @property
    def maximalFraction(self):
        return self._maximalFraction

    def __hash__(self):
        return hash(self.name)

    @abstractmethod
    def __call__(self, *args, **kwargs) -> tuple:
        pass

    def tune(self, population : list):
        pass

    def prepare(self):
        pass

    def standby(self):
        pass


class VOFailed(Exception):
    pass
