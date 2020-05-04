"""
USPEX.Common.Atomistic.CrystalPool
==================================

Contains configuration of Crystal space

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

from ..SystemPool import SystemPool
from .ConvexHull import ConvexHull

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

    MAX_FORMATION_ENERGY = 0.5
    DEFAULT_FITNESS = [('formationEnergy', 'min')]

    def __init__(self, config):
        """
        Initializes the class.

        :type config: :class:`~USPEX.Common.Config.Config` or descendant
        :param config: describes the chemical compositions configuration space.
        """
        super().__init__(config)
        self._convexHull = ConvexHull(self.config)
        self.extendedConvexHull = []

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
