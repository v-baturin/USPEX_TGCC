import unittest
import asyncio
import os


from ..PopulationProcessor import PopulationProcessor, Stages


class Stage1:

    def __init__(self, tag):
        self.tag = tag

    async def run(self, source, sink):
        sink.update(source)
        sink[f'result_{self.tag}'] = f'{self.tag}_Hello!'


class Stage2:

    def __init__(self, tag):
        self.tag = tag

    async def run(self, source, sink):
        sink.update(source)
        sink[f'result_{self.tag}'] = f'{self.tag}_Buy!'


Stages.registerStage('stage1', Stage1)
Stages.registerStage('stage2', Stage2)


class PopulationProcessor_Test(unittest.TestCase):

    def test_life(self):
        stages = [{'stageType': 'stage1', 'tag': 1},
                  {'stageType': 'stage1', 'tag': 2},
                  {'stageType': 'stage1', 'tag': 3}]
        population, sc = PopulationProcessor.initializePopulation('USPEX_stages', [{'ID': i} for i in range(20)])
        asyncio.get_event_loop().run_until_complete(PopulationProcessor.processPopulation(stages, population, 10,
                                                                                          saveCallback=sc))
        population = [system[-1] for system in population.values()]

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
        population, sc = PopulationProcessor.initializePopulation('USPEX_stages', [{'ID': i} for i in range(20)])
        asyncio.get_event_loop().run_until_complete(PopulationProcessor.processPopulation(stages, population, 10,
                                                                                          saveCallback=sc))
        population = [system[-1] for system in population.values()]
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
