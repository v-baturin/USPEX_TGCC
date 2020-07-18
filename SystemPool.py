"""
USPEX.Common.SystemPool
=======================

Contains configuration of such space, parameters of what we are searching for

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""


import logging
logger = logging.getLogger(__name__)


class SystemPool(object):
    """
    This class contains configuration of such space, parameters of what are we searching for,
    list of systems already studied in the search, current result of the search.

    This class is universal for all kind of targets and should be inherited for particular targets
    and its methods overridden.

    :cvar DEFAULT_FITNESS:
        list of tuples describing default optimization type for this pool of systems.
        Each tuple consist of name of attribute and 'min' or 'max' modifier. Class variable.
        For default implementation this list is empty.
    :ivar config:
        link to implementation of :class:`~USPEX.Common.Config.Config` interface.
    :ivar best:
        list of currently best known systems.
    :ivar uniqueSystems:
        list of all currently studied systems.
    """

    def __init__(self):
        """
        Initializes the class.

        :type config: :class:`~USPEX.Common.Config.Config` or descendant
        :param config: describes the chemical compositions configuration space.
        """
        self.uniqueSystems = []
        self.allSystems = {}
        self._newID = 0

    def update(self, population: list):
        """
        Update information about target space in current search.

        :type population: list of :class:`~USPEX.Common.System.System` descendants
        :param population: list of systems which allows to update our knowledge about target space.
        """

        self.best = {}

        logger.info('Updating target: list of unique systems.')
        uniqueIDs = [system.ID for system in self.uniqueSystems]
        for system in population:
            if system.ID not in uniqueIDs:
                logger.debug('add new system %d to list of unique systems' % system.ID)
                uniqueIDs.append(system.ID)
                self.uniqueSystems.append(system)

    def newFoundSystems(self, population: list):
        """
        Determines new found systems in population.

        :type population: list of :class:`~USPEX.Common.System.System` descendants
        :param population: list of systems which allows to update our knowledge about target space.
        :rtype: list
        :return: new found systems.
        """
        logger.info('Determine new systems.')
        uniqueIDs = [system.ID for system in self.uniqueSystems]
        newFoundSystems = []
        for system in population:
            if system.ID not in uniqueIDs:
                logger.debug('found new system %d' % system.ID)
                newFoundSystems.append(system)
                uniqueIDs.append(system.ID)
        return newFoundSystems

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

    def assignID(self, system):
        """
        Assign ID to system.

        :type system: :class:`~USPEX.Common.Atomistic.AtomicStructure.AtomicStructure` descendant
        :param system: system to be labeled with ID.
        """
        system.ID = self._newID
        self._newID += 1
        self.allSystems[system.ID] = system
