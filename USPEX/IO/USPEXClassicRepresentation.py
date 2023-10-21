import matplotlib
import matplotlib.pyplot as plt

from collections import Counter
from pathlib import Path
from .formatters import createHeader_wrap

matplotlib.use('Agg')


class USPEXClassicRepresentation(object):
    def __init__(self, RES_FOLDER: Path, **params):
        self.RES_FOLDER = RES_FOLDER

    def presentFractions(self, optimizer):
        allOperators = set()
        allAmountsAndTotals = []
        for generation in optimizer.pool.generations:
            population = generation['allSystems']
            amounts = Counter()
            for system in population:
                amounts[system['.howCome.origin']] += 1
            total = sum(amounts.values())
            allOperators.update(amounts.keys())
            allAmountsAndTotals.append((amounts, total))

        operatorsFracs = {operator : [amounts[operator]/total for amounts, total in allAmountsAndTotals]
                          for operator in allOperators}

        plt.clf()
        for operator, fracs in operatorsFracs.items():
            plt.plot(fracs, label = operator)
        plt.legend()
        self.RES_FOLDER.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.RES_FOLDER/'VarOperators.svg')


    @staticmethod
    def getParametersBlock(selection) -> list:
        '''

        :return: list of strings
        '''

        header = createHeader_wrap(['Block for evolutionary algorithm'], 'center')
        header.append('')
        header.append(f'    Initial Population Size:    {selection.initialPopSize}')
        header.append(f'    General Population Size:    {selection.popSize}')
        header.append('')

        return header

    @staticmethod
    def getPopulationCreationBlock(population, optimizer, targetRepresentation) -> list:
        block = []
        if not optimizer.createPopulation.globalParentsPool:
            block.append('     Best and diverse structures from previous generation')
            mostDiverseTable = targetRepresentation.getNewSystemsTable()
            for system in optimizer.createPopulation.getMostDiverse():
                mostDiverseTable.update(system['ID'], system)
            block.append(mostDiverseTable.table.get_string())

        amounts = Counter()
        for system in population:
            amounts[system['.howCome.origin']] += 1
        seedsAmount = amounts.pop('Seeds') if 'Seeds' in amounts else 0
        total = sum(amounts.values())

        block += ['    Variation Operators (amount and fraction)',
               *(f'      {howCome:20}:    {amount:4}, {amount/total:4.2}' for howCome, amount in amounts.items()),
                 f'      Seeds               :    {seedsAmount:4}'
        ]
        return block

    @staticmethod
    def applyPresetOutputParameters(optimizer, output):
        return output
