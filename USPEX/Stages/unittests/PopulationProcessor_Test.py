import unittest
import asyncio
import os


from ..PopulationProcessor import PopulationProcessor, Stages
from ...Optimizers.PoolEntry import PoolEntry, EntryFlavour, Pool, FlavourFactory


class Stage1:

    def __init__(self, tag, source, target=None):
        self.tag = tag
        self.source = source

    async def run(self, system):
        system.setProperty('result', f'{self.tag}_Hello!', suffix=self.tag)
        system.setProperty('isBad', False, suffix=self.tag)


class Stage2:

    def __init__(self, tag, source, target=None):
        self.tag = tag
        self.source = source

    async def run(self, system):
        system.setProperty('result', f'{self.tag}_Buy!', suffix=self.tag)
        system.setProperty('isBad', False, suffix=self.tag)


Stages.registerStage('stage1', Stage1)
Stages.registerStage('stage2', Stage2)


class PopulationProcessor_Test(unittest.TestCase):

    def setUp(self) -> None:
        PoolEntry.createEngine(":memory:")

    def test_life(self):
        stages = [{'stageType': 'stage1', 'tag': '1', 'source': 'origin'},
                  {'stageType': 'stage1', 'tag': '2', 'source': '1'},
                  {'stageType': 'stage1', 'tag': '3', 'source': '2'}]
        population = Pool.newPool(FlavourFactory({}), {})
        for i in range(20):
            population.newEntry(EntryFlavour(**{'.howCome': None, '.parent': None, '.isBad': False}))
        asyncio.get_event_loop().run_until_complete(PopulationProcessor.processPopulation(stages, population, 10))

        for ID in population.getIDs():
            system = population.getEntry(ID)
            self.assertEqual(system['.result.1'], '1_Hello!')
            self.assertEqual(system['.result.2'], '2_Hello!')
            self.assertEqual(system['.result.3'], '3_Hello!')
        stages = [{'stageType': 'stage2', 'tag': '1', 'source': 'origin'},
                  {'stageType': 'stage2', 'tag': '2', 'source': '1'},
                  {'stageType': 'stage2', 'tag': '3', 'source': '2'},
                  {'stageType': 'stage2', 'tag': '4', 'source': '3'},
                  {'stageType': 'stage2', 'tag': '5', 'source': '4'},
                  {'stageType': 'stage2', 'tag': '6', 'source': '5'}]
        asyncio.get_event_loop().run_until_complete(PopulationProcessor.processPopulation(stages, population, 10))
        for ID in population.getIDs():
            system = population.getEntry(ID)
            self.assertEqual(system['.result.1'], '1_Hello!')
            self.assertEqual(system['.result.2'], '2_Hello!')
            self.assertEqual(system['.result.3'], '3_Hello!')
            self.assertEqual(system['.result.4'], '4_Buy!')
            self.assertEqual(system['.result.5'], '5_Buy!')
            self.assertEqual(system['.result.6'], '6_Buy!')


if __name__ == '__main__':
    unittest.main()
