import os
import numpy as np
from copy import copy
from datetime import datetime

from ..Presets import presetOutput
from .formatters import createHeader, createHeader_wrap


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
    def __init__(self, optimizerInstance, stages: list, numParallelCalcs: int, path: str = os.getcwd(), output = None, **kwargs):
        self.RES_FOLDER = newResFolderName(path)
        self.OUTPUT_FILE = os.path.join(self.RES_FOLDER, 'OUTPUT.txt')
        self.numStages = len(stages)
        self.numParallelCalcs = numParallelCalcs

        if type(optimizerInstance).__name__ == 'GlobalOptimizer':
            if type(optimizerInstance.createPopulation).__name__ == 'USPEXClassic':
                from .USPEXOutput import USPEXClassicRepresentation, getSelectionConfigRepresentation
                self.presentInfo = USPEXClassicRepresentation(self.RES_FOLDER)
            else:
                raise RuntimeError('Unknown engine type in output initialization.')
            if optimizerInstance.target.name == 'Crystal':
                if output is None:
                    compositionSpace = optimizerInstance.target.utilities.compositionSpace
                    if compositionSpace.minAt == compositionSpace.maxAt:
                        output = presetOutput['CrystalFixComp']
                    else:
                        output = presetOutput['CrystalVarComp']

                self.columns = output['columns']
                from .Crystal.CrystalSystemRepresentation import SystemsTable, CrystalSystemRepresentation
                self.presentSystems = CrystalSystemRepresentation(self.RES_FOLDER, self.numStages, **output)
                from .Crystal.CrystalConfigRepresentation import getTargetConfigRepresentation
                from .Crystal.CrystalPoolRepresentation import CrystalPoolRepresentation
                self.presentOptimizer = CrystalPoolRepresentation(self.RES_FOLDER, **output)

                self.SystemsTable = SystemsTable
                self.getTargetConfigRepresentation = getTargetConfigRepresentation
                self.getSelectionConfigRepresentation = getSelectionConfigRepresentation
            else:
                raise RuntimeError('Unknown target type in output initialization.')
        else:
            raise RuntimeError('Unknown optimizer type in output initialization.')

    def presentOutput(self, populations, optimizer, printDate=True):
        os.makedirs(os.path.dirname(self.OUTPUT_FILE), exist_ok=True)

        header = []
        # Date:
        if printDate:
            formatted_rows = createHeader_wrap(['Job started at ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                                               'center')
            formatted_rows.append('')
            header += formatted_rows

        # header += self.getSelectionConfigRepresentation(selectionConfig)
        # header += self.getTargetConfigRepresentation(targetConfig)

        row = ''
        row += '* There are %d local relaxation steps for each individual structure: *\n' % self.numStages
        # TODO write information about each step
        #     row += '%4s  %-12s  %12s\n' % ('Step', 'Abinitio Code', 'K-resolution')
        header.append(row)

        row = ''
        # TODO write information about submission: local/remote, task manager
        row += '%d parallel calculations are performed simultaneously.\n' % self.numParallelCalcs
        header.append(row)

        output = header
        # ---------------------------------------------------------------------------
        # Write everything to the file:

        for generation, population in enumerate(populations):
            output.append(' Generation {0:4d}'.format(generation))
            table = self.SystemsTable(self.columns)
            for system in population:
                table.update(system['ID'], system, optimizer.fitness)
            output.append(table.table.get_string())

        with open(self.OUTPUT_FILE, 'w') as f:
            for i in output:
                f.write(i + '\n')
