"""
USPEX.Atomistic.IonDistances
============================
"""

import numpy as np
from itertools import combinations_with_replacement

from .VolumeEstimator import VolumeEstimator


class IonDistances():
    """
    Class providing tunable way to calculate minimal allowed distances between atoms.
    """
    def __init__(self, volumeType=0, **kwargs):
        """

        :param volumeType:
            range 0 to 1, 0 corresponds to pure atomic environment for
            volume estimation, 1 to pure molecular one.
            Molecular environment is less dense.
            Intermediate value is a coefficient for molecular environment
            in linear combination of the two.
        :param kwargs: {'<symbol> <symbol>' : <amount>} map from pair of symbols to minimal distance between them.

        """
        self.volumeEstimator = VolumeEstimator(volumeType)

        self._distances = {}
        for key, value in kwargs.items():
            assert isinstance(key, str)
            assert np.isfinite(value)
            s1, s2 = key.split(' ')
            self._distances[(s1,s2)] = value

    def getDistances(self, symbols, pressure):
        """
        For given array of symbols generates matrix of minimal distances.
        If minimal distance for pair of symbols is not predefined calculates it using volumeUtility.

        :param symbols: N array of symbols
        :param pressure: external pressure.

        :return: N*N array of minimal distances.
        """
        symbols = [symbol.short_name for symbol in symbols]
        uniqueSimbols = np.unique(symbols)
        minDistMatrix = {}
        radii = {symbol: self.volumeEstimator.calcAtomVolume(symbol, pressure) ** (1.0 / 3.0)
                 for symbol in uniqueSimbols}
        for s1, s2 in combinations_with_replacement(uniqueSimbols, 2):
            if (s1, s2) in self._distances:
                minDistMatrix[(s2, s1)] = minDistMatrix[(s1, s2)] = self._distances[(s1, s2)]
            elif (s2, s1) in self._distances:
                minDistMatrix[(s1, s2)] = minDistMatrix[(s2, s1)] = self._distances[(s2, s1)]
            elif self.volumeEstimator.volumeType == 0:
                minDistMatrix[(s1, s2)] = minDistMatrix[(s2, s1)] = min(0.22 * (radii[s1] + radii[s2]), 1.2)
            else:
                minDistMatrix[(s1, s2)] = minDistMatrix[(s2, s1)] = 0.45 * (radii[s1] + radii[s2])

        N = len(symbols)
        mDM = np.zeros((N, N), dtype=float)
        for i, j in combinations_with_replacement(range(N), 2):
            s1 = symbols[i]
            s2 = symbols[j]
            mDM[i, j] = mDM[j, i] = minDistMatrix[(s1, s2)]
        return mDM
