import os
from os.path import join as pj
from datetime import datetime

from .formatters import createHeader, createHeader_wrap
from .InputParser import write


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


def newResFolderName(path: str) -> str:
    toCreate = True
    folderNum = 0
    resFolder = None

    while toCreate:
        folderNum += 1
        resFolder = os.path.join(path, f'results{folderNum}')
        if not os.path.isdir(resFolder):
            toCreate = False
    return resFolder


class OutputRepresentation(object):
    PARAMETERS_FILENAME = 'parameters.uspex'

    def __init__(self, optimizerInstance, path: str = os.getcwd(), **params):
        self.RES_FOLDER = newResFolderName(path)
        self.OUTPUT_FILE = os.path.join(self.RES_FOLDER, 'OUTPUT.txt')
        self.numStages = len(params['stages'])
        self.numParallelCalcs = params['numParallelCalcs']
        self.numGenerations = params['numGenerations']
        self.stopCrit = params['stopCrit']

        if 'output' not in params:
            output = {}
            params['output'] = output
        else:
            output = params['output']

        if type(optimizerInstance).__name__ == 'GlobalOptimizer':
            if type(optimizerInstance.createPopulation).__name__ == 'USPEXClassic':
                from .USPEXClassicRepresentation import USPEXClassicRepresentation
                output.update(USPEXClassicRepresentation.applyPresetOutputParameters(optimizerInstance, output))
                self.selectionRepresentation = USPEXClassicRepresentation(self.RES_FOLDER, **output)
            else:
                raise RuntimeError('Unknown engine type in output initialization.')
            if optimizerInstance.target.name == 'Crystal':
                from .CrystalRepresentation import CrystalRepresentation
                output.update(CrystalRepresentation.applyPresetOutputParameters(optimizerInstance, output))
                self.targetRepresentation = CrystalRepresentation(self.RES_FOLDER, **output)
            else:
                raise RuntimeError('Unknown target type in output initialization.')
        else:
            raise RuntimeError('Unknown optimizer type in output initialization.')
        os.makedirs(self.RES_FOLDER, exist_ok=True)
        write(pj(self.RES_FOLDER, self.PARAMETERS_FILENAME), {'main': params})

    def presentSystems(self, systems: dict, optimizer):
        return self.targetRepresentation.presentSystems(systems, optimizer, self.numStages)

    def presentOutput(self, populations, optimizers, optimizer, printDate=True):
        os.makedirs(os.path.dirname(self.OUTPUT_FILE), exist_ok=True)

        # Print the header to the log so it's clear that we execute USPEX:
        output = createHeader('Evolutionary Algorithm Code for Structure Prediction')

        # Date:
        if printDate:
            output += createHeader_wrap([f'Output written {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'], 'center')
            output.append('')

        # Cite:
        text = [
            'Please cite the following suggested papers',
            'when you publish the results obtained from USPEX:',
        ]
        output += createHeader_wrap(text)

        output += createHeader_wrap([text.rstrip() for text in GENERAL_PAPERS.split('\n')], 'left')

        formatted_rows = createHeader_wrap(['Block for generations controller'], 'center')
        formatted_rows.append('')
        formatted_rows.append(f'    Number of Generations  :    {self.numGenerations}')
        formatted_rows.append(f'    Halting criteria       :    {self.stopCrit}')
        formatted_rows.append('')

        output += formatted_rows

        output += self.selectionRepresentation.getParametersBlock(optimizer.createPopulation)
        output += self.targetRepresentation.getParametersBlock(optimizer.target)

        output += createHeader_wrap(['Ab initio calculations'], 'center')

        row = ''
        row += '* There are %d local relaxation steps for each individual structure: *\n' % self.numStages
        # TODO write information about each step
        #     row += '%4s  %-12s  %12s\n' % ('Step', 'Abinitio Code', 'K-resolution')
        output.append(row)

        row = ''
        # TODO write information about submission: local/remote, task manager
        row += '%d parallel calculations are performed simultaneously.\n' % self.numParallelCalcs
        output.append(row)


        output += createHeader_wrap(['Generations block'], 'center')

        for generation, population in enumerate(populations):
            output.append(' Generation {0:4d}'.format(generation))
            output += self.selectionRepresentation.getPopulationCreationBlock(population)
            output.append('    Optimization results')
            table = self.targetRepresentation.getNewSystemsTable()
            for system in population:
                table.update(system['ID'], system, optimizer.fitness)
            output.append(table.table.get_string())
            output += self.targetRepresentation.getPopulationSummaryBlock(population, optimizer)
            output.append('')

        # ---------------------------------------------------------------------------
        # Write everything to the file:
        with open(self.OUTPUT_FILE, 'w') as f:
            for i in output:
                f.write(i + '\n')

        if populations:
            self.selectionRepresentation.presentFractions(populations)
        if optimizers:
            self.targetRepresentation.presentOptimizer(optimizers, optimizer)
