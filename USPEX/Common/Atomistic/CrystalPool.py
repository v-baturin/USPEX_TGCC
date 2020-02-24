from ..SystemPool import SystemPool
from .ConvexHull import ConvexHull
from .Fingerprints.cosine_distance import cosine_distance
from .Fingerprints.Fingerprints import Fingerprints

import logging


logger = logging.getLogger(__name__)


class CrystalPool(SystemPool):

    MAX_FORMATION_ENERGY = 0.5
    DEFAULT_FITNESS = [('formationEnergy', 'min')]

    def __init__(self, config):
        super().__init__(config)
        self._convexHull = ConvexHull(self.config)
        self.extendedConvexHull = []


    def update(self, population : list):
        '''
        :param population: list of a systems.
        '''
        super().update(population)
        logger.info('Updating target: convex hull.')
        for system in population:
            if self._convexHull[system] < 0:
                self._convexHull.add(system)

        for system in self.uniqueSystems:
            if self._convexHull[system] < self.MAX_FORMATION_ENERGY:
                    self.extendedConvexHull.append(system)


    def cleanDuplicates(self, population : list):
        logger.info('Looking for duplicates.')
        cleanedPopulation = []
        for system in population:
            system.f = Fingerprints(system, **self.config.fingerprints)
            logger.debug(f'checking if system {system} is new')
            for ref_system in self.uniqueSystems + cleanedPopulation:
                if system.get_chemical_formula() == ref_system.get_chemical_formula() and \
                   cosine_distance(system.f.fingerprint, ref_system.f.fingerprint, system.f.weight) < self.config.fingerprints['tolerance']:
                    logger.debug(f'system {system} coincides with system {ref_system} found earlier')
                    system = ref_system
                    break

            if not system.isBad:
                cleanedPopulation.append(system)

        population.clear()
        population.extend(cleanedPopulation)

    def formationEnergy(self, system):
        return self._convexHull[system]
