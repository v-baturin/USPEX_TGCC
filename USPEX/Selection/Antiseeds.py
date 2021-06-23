import numpy as np
from itertools import combinations

ANTISEEDS_MAX = 0.005
ANTISEEDS_SIGMA = 0.001


class Antiseeds:

    def __init__(self, max = ANTISEEDS_MAX, sigma = ANTISEEDS_SIGMA):
        self.max = max
        self.sigma = sigma

    def payPenalties(self, population, pool, fingerprintUtility):
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
            if 'antiseeds.corrections' in system:
                for ref_system in population:
                    dist = fingerprintUtility.dist(ref_system, system)
                    system['antiseeds.corrections'] += self.max * np.exp(-dist**2/(2*sigma**2))
            else:
                system['antiseeds.corrections'] = 0
                for ref_system in pool:
                    dist = fingerprintUtility.dist(ref_system, system)
                    system['antiseeds.corrections'] += self.max * np.exp(-dist**2/(2*sigma**2))

    def corrections(self, system : dict):
        return system['antiseeds.corrections'] if 'antiseeds.corrections' in system else 0
