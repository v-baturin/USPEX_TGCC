import unittest
import asyncio
import os
from copy import copy


from .. import Stages
from ..PopulationProcessor import PopulationProcessor


PopulationProcessor.setStages(Stages)


class Stage1:

    def __init__(self, tag):
        self.tag = tag

    async def run(self, system):
        results = copy(system)
        results[f'result_{self.tag}'] = f'{self.tag}_Hello!'
        return results


class Stage2:

    def __init__(self, tag):
        self.tag = tag

    async def run(self, system):
        results = copy(system)
        results[f'result_{self.tag}'] = f'{self.tag}_Buy!'
        return results


Stages.registerStage('stage1', Stage1)
Stages.registerStage('stage2', Stage2)


class PopulationProcessor_Test(unittest.TestCase):

    def test_life(self):
        stages = [{'stageType': 'stage1', 'tag': 1},
                  {'stageType': 'stage1', 'tag': 2},
                  {'stageType': 'stage1', 'tag': 3}]
        population = [{'ID': i} for i in range(20)]
        populationProcessor1 = PopulationProcessor(tag='stages', stages=stages, inputKey='population', numParallelCalcs=10)
        population = asyncio.get_event_loop().run_until_complete(populationProcessor1.run(dict(ID='USPEX', population=population)))['population']
        for system in population:
            self.assertEqual(system['result_1'], '1_Hello!')
            self.assertEqual(system['result_2'], '2_Hello!')
            self.assertEqual(system['result_3'], '3_Hello!')
        stages = [{'stageType': 'stage2', 'tag': 1},
                  {'stageType': 'stage2', 'tag': 2},
                  {'stageType': 'stage2', 'tag': 3},
                  {'stageType': 'stage2', 'tag': 4},
                  {'stageType': 'stage2', 'tag': 5},
                  {'stageType': 'stage2', 'tag': 6}]
        populationProcessor2 = PopulationProcessor(tag='stages', stages=stages, inputKey='population', numParallelCalcs=10)
        population = asyncio.get_event_loop().run_until_complete(populationProcessor2.run(dict(ID='USPEX', population=population)))['population']
        for system in population:
            self.assertEqual(system['result_1'], '1_Hello!')
            self.assertEqual(system['result_2'], '2_Hello!')
            self.assertEqual(system['result_3'], '3_Hello!')
            self.assertEqual(system['result_4'], '4_Buy!')
            self.assertEqual(system['result_5'], '5_Buy!')
            self.assertEqual(system['result_6'], '6_Buy!')
        os.remove('USPEX_stages.dump')
        os.remove('USPEX_stages.dump.back')


if __name__ == '__main__':
    unittest.main()
