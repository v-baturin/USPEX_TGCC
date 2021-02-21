import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .formatters import createHeader, createHeader_wrap

GENERAL_PAPERS = '''\
Oganov A.R., Glass C.W. (2006)
Crystal structure prediction using evolutionary algorithms:
principles and applications.
J. Chem. Phys. 124, 244704

Oganov A.R., Stokes H., Valle M. (2011)
How evolutionary crystal structure prediction works - and why.
Acc. Chem. Res. 44, 227-237

Lyakhov A.O., Oganov A.R., Stokes H., Zhu Q. (2013)
New developments in evolutionary structure prediction algorithm USPEX.
Comp. Phys. Comm., 184, 1172-1182\
'''


def getSelectionConfigRepresentation(config) -> list:
    '''

    :return: list of strings
    '''

    header = []
    # Create a header automatically:
    description = 'Evolutionary Algorithm Code for Structure Prediction'

    # Print the header to the log so it's clear that we execute USPEX:
    formatted_rows = createHeader(description)


    header += formatted_rows

    text = ['Block for evolutionary algorithm']
    formatted_rows = createHeader_wrap(text, 'center')
    formatted_rows.append('')


    #row =  '    Number of Generations  :    %4d\n' % NUM_GENERATIONS
    # row = '    Initial Population Size:    %4d\n' % config['initialPopSize']
    row = '    General Population Size:    %4d\n' % config['popSize']

    formatted_rows.append(row)
    header += formatted_rows

    # Cite:
    text = [
        'Please cite the following suggested papers',
        'when you publish the results obtained from USPEX:',
    ]
    formatted_rows = createHeader_wrap(text)
    header += formatted_rows

    text = GENERAL_PAPERS.split('\n')

    for i in range(len(text)):
        text[i] = text[i].rstrip()
    formatted_rows = createHeader_wrap(text, 'left')
    header += formatted_rows

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
