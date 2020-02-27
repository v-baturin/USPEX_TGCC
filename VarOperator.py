'''
@file        VarOperator.py
@author:     Pavel Bushlanov
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    paulbush@mail.ru
@date        21 July 2016
@brief       Abstract class for different variation operators
'''


from USPEX.Common.Config import Config

MAX_OUTPUT_SIZE = 8


class VarOperator(object):
    '''

    '''

    name = None
    isActive = True

    def __init__(self, config : Config, pool, initFrac : float, minFrac : float=0.1, maxFrac : float=1.0, maxOutputSize : int=MAX_OUTPUT_SIZE):
        '''

        :param config:
        :param initFrac:
        :param minFrac:
        :param maxFrac:
        '''

        self.config = config
        self.pool = pool
        assert minFrac >= 0.0 and minFrac <= 1.0
        assert maxFrac >= 0.0 and maxFrac <= 1.0
        assert minFrac <= maxFrac
        self._minimalFraction = minFrac
        self._maximalFraction = maxFrac
        self._initialFraction = initFrac
        self._MAX_OUTPUT_SIZE = int(maxOutputSize)

    def __str__(self):
        return self.name

    @property
    def initialFraction(self):
        return self._initialFraction

    @property
    def minimalFraction(self):
        return self._minimalFraction

    @property
    def maximalFraction(self):
        return self._maximalFraction

    @property
    def maxOutputSize(self) -> int:
        return self._MAX_OUTPUT_SIZE

    def __hash__(self):
        return hash(self.name)

    def __call__(self, *args, **kwargs) -> tuple:
        return ()

    def tune(self, population : list):
        pass

    def prepare(self):
        pass

    def standby(self):
        pass


class VOFailed(Exception):
    pass
