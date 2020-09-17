import logging
logger = logging.getLogger(__name__)


import numpy as np
import pandas as pd

from itertools import chain, combinations_with_replacement

from ..ConvexHull import ConvexHull
from ..Fitness import Fitness
from ..SystemPool import SystemPool


# To which dimensionality we project our FP and consider
MAX_DIMENSIONALITY = 7

class GeneralizedConvexHull(ConvexHull):
    '''
    Class for the building generalized convex hull.
    It shows which structures are stable and which are not.

    Create convex hull object:
    >>> convex_hull = GeneralizedConvexHull(systems)

    Extend GCH with new protion of systems:
    >>> convex_hull.extend(systems)

    Convex hull is rebuild on every call to extend due to the projection mechanism
     which is based on previously found stuctures.
    '''
    def __init__(self, systems: list, config, dimensionality : int=None):
        '''
        :param dimensionality: dimension of the FP projection
        :param saved_data: path to file, where will be stored temporary data
        '''
        self.config = config

        # Set of unique chemical symbols
        self._symbols = set(config.chemicalSymbols)

        # If dimension is not set we define it like:
        if dimensionality is not None:
            assert dimensionality > 1  # Must be = 1D (energy) + nD (fingerprints projection)
            self.DIMENSIONALITY = 1+dimensionality  # Energy + FP projection
        else:
            d = 2 + len(self._symbols)
            self.DIMENSIONALITY = np.min([1+d, MAX_DIMENSIONALITY])
        logger.info(f'Dimensionality of GCH = {self.DIMENSIONALITY}')

        self.systems = systems
        if len(self.systems) <= self.DIMENSIONALITY:
            self._df = pd.DataFrame(columns=['argument', 'property', 'height', 'depth'])
            for i, system in enumerate(systems):
                self._df.loc[i] = None, None, 0.0, 0.0
        else:
            pool = SystemPool()
            pool.update(self.systems)
            super().__init__(Fitness(pool, []).calcFitness(('getAbsoluteCHSpace',
                                                                    ('getPrincipalComponents', self.DIMENSIONALITY - 1,
                                                                     ('hstack', ('tabulate', 'fingerprint'))),
                                                                    'enthalpy')))

    @property
    def lower_bound(self):
        return [self.systems[i] for i in super(GeneralizedConvexHull, self).lower_bound]

    @property
    def upper_bound(self):
        return [self.systems[i] for i in super(GeneralizedConvexHull, self).upper_bound]

    @property
    def depth(self):
        return super().depth

    @property
    def height(self):
        return super().height

    def extend(self, systems: list):
        self.systems.extend(systems)
        if len(self.systems) <= self.DIMENSIONALITY:
            self._df = pd.DataFrame(columns=['argument', 'property', 'height', 'depth'])
            for i, system in enumerate(systems):
                self._df.loc[i] = None, None, 0.0, 0.0
        else:
            pool = SystemPool()
            pool.update(self.systems)
            super().__init__(Fitness(pool, []).calcFitness(('getAbsoluteCHSpace',
                                                                    ('getPrincipalComponents', self.DIMENSIONALITY - 1,
                                                                     ('hstack', ('tabulate', 'fingerprint'))),
                                                                    'enthalpy')))
