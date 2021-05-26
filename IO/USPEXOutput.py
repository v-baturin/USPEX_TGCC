import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .formatters import createHeader_wrap


def getSelectionConfigRepresentation(selection) -> list:
    '''

    :return: list of strings
    '''

    header = createHeader_wrap(['Block for evolutionary algorithm'], 'center')
    header.append('')
    header.append(f'    Initial Population Size:    {selection.initialPopSize}')
    header.append(f'    General Population Size:    {selection.popSize}')
    header.append('')

    return header


class USPEXClassicRepresentation(object):
    def __init__(self, RES_FOLDER : str):
        self.RES_FOLDER  = RES_FOLDER

    def __call__(self, info):
        operatorsFracs = {}
        for analysis in info:
            weightsLast, weightsBest = analysis[0]
            weightsSum = np.sum(np.fromiter(weightsLast.values(), dtype=float))
            operators = list(set(weightsLast.keys()) | set(operatorsFracs.keys()))
            if operatorsFracs:
                generation = len(list(operatorsFracs.values()))
            else:
                generation = 0
            for operator in operators:
                if operator not in operatorsFracs:
                    operatorsFracs[operator] = [0.0] * generation
                if operator in weightsLast:
                    operatorsFracs[operator].append(weightsLast[operator] / weightsSum)
                else:
                    operatorsFracs[operator].append(0.0)

        plt.clf()
        for operator, fracs in operatorsFracs.items():
            plt.plot(fracs, label = operator)
        plt.legend()
        os.makedirs(self.RES_FOLDER, exist_ok=True)
        plt.savefig(self.RES_FOLDER + '/VarOperators.svg')
