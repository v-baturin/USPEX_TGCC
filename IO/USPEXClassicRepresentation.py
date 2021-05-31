import os
import numpy as np
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .formatters import createHeader_wrap


class USPEXClassicRepresentation(object):
    def __init__(self, RES_FOLDER : str, **params):
        self.RES_FOLDER  = RES_FOLDER

    def presentFractions(self, populations):
        allOperators = set()
        allAmountsAndTotals = []
        for population in populations:
            amounts = Counter()
            for system in population:
                amounts[system['howCome']] += 1
            total = sum(amounts.values())
            allOperators.update(amounts.keys())
            allAmountsAndTotals.append((amounts, total))

        operatorsFracs = {operator : [amounts[operator]/total for amounts, total in allAmountsAndTotals]
                          for operator in allOperators}

        plt.clf()
        for operator, fracs in operatorsFracs.items():
            plt.plot(fracs, label = operator)
        plt.legend()
        os.makedirs(self.RES_FOLDER, exist_ok=True)
        plt.savefig(self.RES_FOLDER + '/VarOperators.svg')


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
        mostDiverseTable = targetRepresentation.getNewSystemsTable()
        for system in optimizer.createPopulation.getMostDiverse():
            mostDiverseTable.update(system['ID'], system, optimizer.fitness)

        amounts = Counter()
        for system in population:
            amounts[system['howCome']] += 1
        seedsAmount = amounts['Seeds']
        del amounts['Seeds']
        total = sum(amounts.values())

        block = ['     Best and diverse structures form previous generation',
                  mostDiverseTable.table.get_string(),
                  '    Variation Operators (amount and fraction)',
               *(f'      {howCome:20}:    {amount:4}, {amount/total:4.2}' for howCome, amount in amounts.items()),
                 f'      Seeds               :    {seedsAmount:4}'
        ]
        return block

    @staticmethod
    def applyPresetOutputParameters(optimizer, output):
        return output