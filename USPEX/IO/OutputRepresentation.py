from datetime import datetime
from itertools import zip_longest
from pathlib import Path

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


def newResFolderName(path: str) -> Path:
    toCreate = True
    folderNum = 0
    resFolder = None

    while toCreate:
        folderNum += 1
        resFolder = Path(path)/f'results{folderNum}'
        if not resFolder.is_dir():
            toCreate = False
    return resFolder


class OutputRepresentation(object):
    PARAMETERS_FILENAME = 'parameters.uspex'

    def __init__(self, optimizerInstance, path: str = './', **params):
        self.RES_FOLDER = newResFolderName(path)
        self.OUTPUT_FILE = self.RES_FOLDER/'OUTPUT.txt'
        self.stages = params['stages']
        self.numParallelCalcs = params['numParallelCalcs']
        self.numGenerations = params['numGenerations']
        self.stopCrit = params['stopCrit']

        if 'output' not in params:
            output = {}
            params['output'] = output
        else:
            output = params['output']

        if type(optimizerInstance).__name__ == 'GlobalOptimizer':
            from .USPEXClassicRepresentation import USPEXClassicRepresentation
            output.update(USPEXClassicRepresentation.applyPresetOutputParameters(optimizerInstance, output))
            self.selectionRepresentation = USPEXClassicRepresentation(self.RES_FOLDER, **output)
            if optimizerInstance.target.name == 'Atomistic':
                from .AtomisticRepresentation import AtomisticRepresentation
                # output = dict(AtomisticRepresentation.applyPresetOutputParameters(optimizerInstance), **output)
                self.targetRepresentation = AtomisticRepresentation(self.RES_FOLDER, **output)
            else:
                raise RuntimeError('Unknown target type in output initialization.')
        elif type(optimizerInstance).__name__ == 'ModelOptimizer':
            self.selectionRepresentation = None
            self.targetRepresentation = None
        else:
            raise RuntimeError('Unknown optimizer type in output initialization.')
        self.RES_FOLDER.mkdir(parents=True, exist_ok=True)
        write(self.RES_FOLDER/self.PARAMETERS_FILENAME, params)

    def presentSystems(self, optimizer):
        if self.targetRepresentation is not None:
            return self.targetRepresentation.presentSystems(optimizer)
        else:
            return None

    def presentOutput(self, optimizer, printDate=True, final=False):
        if self.selectionRepresentation is not None and self.targetRepresentation is not None:
            self.OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

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

            output += self.selectionRepresentation.getParametersBlock(optimizer._createPopulation)
            output += self.targetRepresentation.getParametersBlock(optimizer.target)

            output += createHeader_wrap(['Ab initio calculations'], 'center')

            row = '\n'
            row += f'* There are {len(self.stages)} local relaxation steps for each individual structure: *\n'
            row += '   Step/Tag     Abinitio Code   K-resolution\n'
            for stage in self.stages:
                kresol = stage['kresol'] if 'kresol' in stage else None
                row += f"{stage['tag']:12}    {stage['type']:12}    {kresol}\n"
            row += '\n'
            row += '%d parallel calculations are performed simultaneously.\n' % self.numParallelCalcs
            row += 'For submission details of each stage see parameters.txt.'
            row += '\n'
            output.append(row)


            output += createHeader_wrap(['Generations block'], 'center')

            for i, generation in enumerate(optimizer.generations):
                population = generation.population
                output.append(' Generation {0:4d}'.format(i))
                output += self.selectionRepresentation.getPopulationCreationBlock(population, optimizer,
                                                                                  self.targetRepresentation)
                output.append('    Optimization results')
                table = self.targetRepresentation.getNewSystemsTable()
                for ID in population.getIDs():
                    table.update(ID, optimizer.allSystems.getEntry(ID))
                output.append(table.table.get_string())
                output += self.targetRepresentation.getPopulationSummaryBlock(population, optimizer)
                output.append('')


            if final:
                table = self.targetRepresentation.getNewSystemsTable()
                for ID in optimizer.best:
                    table.update(ID, optimizer.allSystems.getEntry(ID))
                output += createHeader_wrap(['Calculation results'], 'center')
                output.append(table.table.get_string())

            # ---------------------------------------------------------------------------
            # Write everything to the file:
            with open(self.OUTPUT_FILE, 'w') as f:
                for i in output:
                    f.write(i + '\n')

            self.selectionRepresentation.presentFractions(optimizer)
            self.targetRepresentation.presentOptimizer(optimizer)
