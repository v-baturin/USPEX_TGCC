import numpy as np
from itertools import combinations

from ..DataModel.Pool import Pool

ANTISEEDS_MAX = 0.005
ANTISEEDS_SIGMA = 0.001


class Antiseeds:
    """
    Utility which calculates and store penalties for systems.
    Such penalties applied not only to some system itself but to all its neighbours with gaussian distribution.
    """

    def __init__(self, prop: str = 'corrections', max: float = ANTISEEDS_MAX, sigma: float = ANTISEEDS_SIGMA):
        """
        :param max: height of gaussian distribution.
        :param sigma: width of gaussian distribution.
        """
        self.prop = prop
        self.max = max
        self.sigma = sigma

    def payPenalties(self, population: Pool, pool: Pool, metric):
        """
        Calculates and stores penalties.
        :param population: list of systems to be penalized. This systems will be in centers of gaussian distributions.
        :param pool: list of all systems.
        All this systems will get penalties depending on their distance from systems in *popuation* list.
        :param metric: utility providing **dist** method which calculates distance between systems.
        """
        suffix = metric.suffix
        population = [population.getEntry(ID) for ID in population.getIDs()]
        pool = [pool.getEntry(ID) for ID in pool.getIDs()]
        comb = list(combinations(population, 2))
        sigma = self.sigma*(np.sum(list(metric.dist(s1, s2) for s1, s2 in comb)) / len(comb) if comb else 1)
        for system in pool:
            correction = system.getProperty(self.prop, extension='antiseeds', suffix=suffix)
            dists = np.fromiter((metric.dist(ref, system) for ref in population), dtype=float)
            correction += self.max * np.sum(np.exp(-dists ** 2 / (2 * sigma ** 2)))
            system.setProperty(self.prop, correction, extension='antiseeds', suffix=suffix)
