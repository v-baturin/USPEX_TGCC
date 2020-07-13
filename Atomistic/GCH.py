import logging
logger = logging.getLogger(__name__)


import numpy as np

from itertools import chain, combinations_with_replacement

from ..ConvexHull import ConvexHull


# To which dimensionality we project our FP and consider
MAX_DIMENSIONALITY = 7


class GeneralizedConvexHull(ConvexHull):
    '''
    Class for the building generalized convex hull.
    It shows which structures are stable and which are not.

    Create empty convex hull object:
    >>> convex_hull = GeneralizedConvexHull(config)

    Every single generation GCH must be rebuild in the following way:
    >>> convex_hull.clean()
    >>> convex_hull.extend(uniqueSystems + population)

    This is due to the projection mechanism which is based on previously found stuctures.
    '''

    def __init__(self, config, dimensionality : int=None):
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

        self.orig_systems = []
        super().__init__()

    def _flatten_fp(self, fingerprint):
        '''
        Flatten fingerprint into a long list by the follows rule:
        :param fp_value: value of the FP
        '''
        N = fingerprint.size
        fp_value = fingerprint.value
        fp_data = []
        for p in combinations_with_replacement(self._symbols, 2):
            fp_data.append(fp_value[p] if p in fp_value else -np.ones(N))
        return list(chain.from_iterable(fp_data))

    def extend(self, systems):
        '''
        :param systems: N * (ID + structure with enthalpy)
        '''

        self.orig_systems.extend(systems)
        if len(self.orig_systems) <= self.DIMENSIONALITY:  # not enough points to build GCH
            for i, system in enumerate(self.orig_systems):
                self._df.loc[i] = system, None, None, 0.0, 0.0
            return

        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.DIMENSIONALITY-1)
        transformed = pca.fit_transform(np.array([self._flatten_fp(x.fingerprint) for x in self.orig_systems]))

        systems_dicts = []
        for system, coord in zip(self.orig_systems, transformed):
            composition = self.config.numBlocks(system.composition)
            systems_dicts.append({'argument': coord, 'property': system.enthalpy/np.sum(composition)})

        super().extend(systems_dicts)
