import numpy as np
from itertools import combinations_with_replacement


class IonDistances():
    def __init__(self, **kwargs):
        self._distances = kwargs
        for key, value in kwargs.items():
            assert isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], str) and isinstance(key[1], str)
            assert np.isfinite(value)

    def getDistances(self, symbols, volumeUtility):
        symbols = [symbol.short_name for symbol in symbols]
        uniqueSimbols = np.unique(symbols)
        minDistMatrix = {}
        radii = {symbol: volumeUtility.calcAtomVolume(symbol) ** (1.0 / 3.0)
                 for symbol in uniqueSimbols}
        for s1, s2 in combinations_with_replacement(uniqueSimbols, 2):
            if (s1, s2) in self._distances:
                minDistMatrix[(s2, s1)] = minDistMatrix[(s1, s2)] = self._distances[(s1, s2)]
            elif (s2, s1) in self._distances:
                minDistMatrix[(s1, s2)] = minDistMatrix[(s2, s1)] = self._distances[(s2, s1)]
            elif volumeUtility.volumeType == 0:
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
