"""
USPEX.Common.SystemPool
=======================

Contains configuration of such space, parameters of what we are searching for

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""


import logging
logger = logging.getLogger(__name__)

from copy import copy

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
    :ivar uniqueSystems:
        list of all currently studied systems.
    """

    def __init__(self):
        """
        Initializes the class.

        :type config: :class:`~USPEX.Common.Config.Config` or descendant
        :param config: describes the chemical compositions configuration space.
        """
        self.uniqueSystems = ()
        self.allSystems = {}
        self._newID = 0

    def __copy__(self):
        other = SystemPool.__new__(SystemPool)
        other.uniqueSystems = self.uniqueSystems
        other.allSystems = copy(self.allSystems)
        other._newID = self._newID
        return other

    def __hash__(self):
        return hash(tuple(system['ID'] for system in self.uniqueSystems))

    def update(self, population: list):
        """
        Update information about target space in current search.

        :type population: list of :class:`~USPEX.Common.System.System` descendants
        :param population: list of systems which allows to update our knowledge about target space.
        """

        logger.info('Updating target: list of unique systems.')
        uniqueIDs = [system['ID'] for system in self.uniqueSystems]
        uniqueSystems = list(self.uniqueSystems)
        for system in population:
            if system['ID'] not in uniqueIDs:
                logger.debug('add new system %d to list of unique systems' % system['ID'])
                uniqueIDs.append(system['ID'])
                uniqueSystems.append(system)
        self.uniqueSystems = tuple(uniqueSystems)

    def newFoundSystems(self, population: list):
        """
        Determines new found systems in population.

        :type population: list of :class:`~USPEX.Common.System.System` descendants
        :param population: list of systems which allows to update our knowledge about target space.
        :rtype: list
        :return: new found systems.
        """
        logger.info('Determine new systems.')
        uniqueIDs = [system['ID'] for system in self.uniqueSystems]
        newFoundSystems = []
        for system in population:
            if system['ID'] not in uniqueIDs:
                logger.debug('found new system %d' % system['ID'])
                newFoundSystems.append(system)
                uniqueIDs.append(system['ID'])
        return newFoundSystems

    def assignID(self, system):
        """
        Assign ID to system.

        :type system: :class:`~USPEX.Common.Atomistic.AtomicStructure.AtomicStructure` descendant
        :param system: system to be labeled with ID.
        """
        system['ID'] = self._newID
        self._newID += 1
        self.allSystems[system['ID']] = system
