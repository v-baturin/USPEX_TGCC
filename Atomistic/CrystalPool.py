"""
USPEX.Common.Atomistic.CrystalPool
==================================

Contains configuration of Crystal space

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""


from ..SystemPool import SystemPool
from .CompositionSpace import CompositionSpace
from .ConvexHull import ConvexHull
from .Fingerprints.cosine_distance import cosine_distance

import numpy as np
from itertools import combinations
import logging
logger = logging.getLogger(__name__)


class CrystalPool(SystemPool):
    """
    This class contains configuration of Crystal space, list of systems
    already studied in the search, current result of the search.

    :cvar MAX_FORMATION_ENERGY:
        if the formation energy of the system is greater than this value,
        then the system is not added to extendedConvexHull.
    """

    ANTISEEDS_MAX = 0.005
    ANTISEEDS_SIGMA = 0.001
    MAX_FORMATION_ENERGY = 0.5
    DEFAULT_FITNESS = [('formationEnergy', 'min_antiseeds')]

    def __init__(self, **kwargs):
        """
        Initializes the class.
        """
        super().__init__()

        self.compositionSpace = CompositionSpace(**kwargs)
        self._convexHull = ConvexHull(self.compositionSpace)
        self.extendedConvexHull = []
        self._antiseeds = {}

    def update(self, population: list):
        """
        Update information about target space in current search.

        :type population: list of :class:`~USPEX.Common.System.System` descendants
        :param population: list of systems which allows to update our knowledge about target space.
        """
        super().update(population)
        logger.info('Updating target: convex hull.')
        for system in population:
            if self._convexHull[system] < 0:
                self._convexHull.add(system)

        for system in self.uniqueSystems:
            if self._convexHull[system] < self.MAX_FORMATION_ENERGY:
                    self.extendedConvexHull.append(system)

        comb = list(combinations(population, 2))
        if comb:
            sigma = 0
            for system1, system2 in comb:
                dist = cosine_distance(system1.fingerprint, system2.fingerprint,
                                       system1.fingerprintWeights, system2.fingerprintWeights)
                sigma += dist
            sigma /= len(comb)
        else:
            sigma = 1
        sigma *= self.ANTISEEDS_SIGMA
        for system in self.uniqueSystems:
            if system.ID in self._antiseeds:
                self._antiseeds[system.ID].append(sigma)
            else:
                self._antiseeds[system.ID] = [sigma]

    def cleanDuplicates(self, population: list):
        """
        Method for cleaning duplicates.

        :type population: list of :class:`~USPEX.Common.System.System` descendants
        :param population: list of systems which allows to update our knowledge about target space.
        """
        logger.info('Looking for duplicates.')
        cleanedPopulation = []
        for system in population:
            logger.debug(f'checking if system {system} is new')
            for ref_system in self.uniqueSystems + cleanedPopulation:
                if system == ref_system:
                    logger.debug(f'system {system} coincides with system {ref_system} found earlier')
                    system = ref_system
                    break

            if not system.isBad:
                cleanedPopulation.append(system)

        population.clear()
        population.extend(cleanedPopulation)

    def formationEnergy(self, system):
        """
        Returns energy above convex hull of an input system.

        :type system: :class:`~USPEX.Common.System.System`
        :param system: the system of which we want the energy above convex hull.
        :rtype: float
        :return: energy above convex hull.
        """
        return self._convexHull[system]

    def antiseedsCorrection(self, system) -> float:
        correction = 0
        for ref_system in self.uniqueSystems:
            if ref_system.ID in self._antiseeds:
                for sigma in self._antiseeds[ref_system.ID]:
                    dist = cosine_distance(ref_system.fingerprint, system.fingerprint,
                                           ref_system.fingerprintWeights, system.fingerprintWeights)
                    correction += np.exp(-dist**2/(2*sigma**2))
        return self.ANTISEEDS_MAX*correction
