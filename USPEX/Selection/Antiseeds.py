import numpy as np
from itertools import combinations

ANTISEEDS_MAX = 0.005
ANTISEEDS_SIGMA = 0.001


class Antiseeds:
    """
    Utility which calculates and store penalties for systems.
    Such penalties applied not only to some system itself but to all its neighbours with gaussian distribution.
    """

    def __init__(self, max=ANTISEEDS_MAX, sigma=ANTISEEDS_SIGMA, legacy=True, **kwargs):
        """
        :param max: height of gaussian distribution.
        :param sigma: width of gaussian distribution.
        """
        self.max = max
        self.sigma = sigma
        self.legacy = legacy

    def payPenalties(self, population, pool, fingerprintUtility):
        """
        Calculates and stores penalties.
        :param population: list of systems to be penalized. This systems will be in centers of gaussian distributions.
        :param pool: list of all systems.
        All this systems will get penalties depending on their distance from systems in *popuation* list.
        :param fingerprintUtility: utility providing **dist** method which calculates distance between systems.
        """
        suffix = fingerprintUtility.suffix
        comb = list(combinations(population, 2))
        if comb:
            sigma = 0
            for s1, s2 in comb:
                sigma += fingerprintUtility.dist(s1, s2)
            sigma /= len(comb)
        else:
            sigma = 1
        sigma *= self.sigma
        for system in pool:
            if f'antiseeds.corrections.{suffix}' in system:
                for ref_system in population:
                    dist = fingerprintUtility.dist(ref_system, system)
                    correction = system[f'antiseeds.corrections.{suffix}']
                    system.setProperty('corrections', correction + self.max * np.exp(-dist ** 2 / (2 * sigma ** 2)),
                                       prefix='antiseeds', suffix=suffix)
            else:
                system.setProperty('corrections', 0, prefix='antiseeds', suffix=suffix)

    def corrections(self, system):
        """
        For using in **Fitness** infrastructure
        :param system: dictionary describing system.
        :return: retrieve antiseeds penalty of a system.
        """
        return system['antiseeds.corrections'] if 'antiseeds.corrections' in system else 0
