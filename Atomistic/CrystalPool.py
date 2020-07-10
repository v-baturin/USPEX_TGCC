"""
USPEX.Common.Atomistic.CrystalPool
==================================

Contains configuration of Crystal space

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import logging
logger = logging.getLogger(__name__)


from ..SystemPool import SystemPool
from .CompositionSpace import CompositionSpace


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
