import unittest
import asyncio
import os


from ..PopulationProcessor import PopulationProcessor, Stages
from ...Optimizers.PoolEntry import PoolEntry


class Stage1:

    def __init__(self, tag, target=None):
        self.tag = tag

    async def run(self, system):
        system.setProperty('result', f'{self.tag}_Hello!', suffix=self.tag)


class Stage2:

    def __init__(self, tag, target=None):
        self.tag = tag

    async def run(self, system):
        system.setProperty('result', f'{self.tag}_Buy!', suffix=self.tag)


Stages.registerStage('stage1', Stage1)
Stages.registerStage('stage2', Stage2)


class PopulationProcessor_Test(unittest.TestCase):

    def test_life(self):
        stages = [{'stageType': 'stage1', 'tag': '1'},
                  {'stageType': 'stage1', 'tag': '2'},
                  {'stageType': 'stage1', 'tag': '3'}]
        initial = [PoolEntry(ID=i) for i in range(20)]
        population, sc = PopulationProcessor.initializePopulation('USPEX_stages', initial)
        asyncio.get_event_loop().run_until_complete(PopulationProcessor.processPopulation(stages, population, 10,
                                                                                          saveCallback=sc))

        for system in population.values():
            self.assertEqual(system['.result.1'], '1_Hello!')
            self.assertEqual(system['.result.2'], '2_Hello!')
            self.assertEqual(system['.result.3'], '3_Hello!')
        stages = [{'stageType': 'stage2', 'tag': '1'},
                  {'stageType': 'stage2', 'tag': '2'},
                  {'stageType': 'stage2', 'tag': '3'},
                  {'stageType': 'stage2', 'tag': '4'},
                  {'stageType': 'stage2', 'tag': '5'},
                  {'stageType': 'stage2', 'tag': '6'}]
        population, sc = PopulationProcessor.initializePopulation('USPEX_stages', initial)
        asyncio.get_event_loop().run_until_complete(PopulationProcessor.processPopulation(stages, population, 10,
                                                                                          saveCallback=sc))
        for system in population.values():
            self.assertEqual(system['.result.1'], '1_Hello!')
            self.assertEqual(system['.result.2'], '2_Hello!')
            self.assertEqual(system['.result.3'], '3_Hello!')
            self.assertEqual(system['.result.4'], '4_Buy!')
            self.assertEqual(system['.result.5'], '5_Buy!')
            self.assertEqual(system['.result.6'], '6_Buy!')
        os.remove('USPEX_stages.dump')
        os.remove('USPEX_stages.dump.back')


if __name__ == '__main__':
    unittest.main()
