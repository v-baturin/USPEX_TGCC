import unittest
import asyncio
import os


from ..PopulationProcessor import PopulationProcessor


class Stage1:

    def __init__(self, tag):
        self.tag = tag

    async def run(self, system):
        system[f'result_{self.tag}'] = f'{self.tag}_Hello!'


class Stage2:

    def __init__(self, tag):
        self.tag = tag

    async def run(self, system):
        system[f'result_{self.tag}'] = f'{self.tag}_Buy!'


class PopulationProcessor_Test(unittest.TestCase):

    def test_life(self):
        stages = [Stage1('1'), Stage1('2'), Stage1('3')]
        population = [{'ID': i} for i in range(20)]
        populationProcessor1 = PopulationProcessor(tag='stages', stages=stages, inputKey='population', numParallelCalcs=10)
        asyncio.get_event_loop().run_until_complete(populationProcessor1.run(dict(ID='USPEX', population=population)))
        for system in population:
            self.assertEqual(system['result_1'], '1_Hello!')
            self.assertEqual(system['result_2'], '2_Hello!')
            self.assertEqual(system['result_3'], '3_Hello!')
        stages = [Stage2('1'), Stage2('2'), Stage2('3'), Stage2('4'), Stage2('5'), Stage2('6')]
        populationProcessor2 = PopulationProcessor(tag='stages', stages=stages, inputKey='population', numParallelCalcs=10)
        asyncio.get_event_loop().run_until_complete(populationProcessor2.run(dict(ID='USPEX', population=population)))
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
