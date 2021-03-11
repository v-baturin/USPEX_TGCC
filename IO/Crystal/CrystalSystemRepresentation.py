import os
import io
import shutil
import matplotlib
import numpy as np

from ase.io import write
from os.path import join as pj
from prettytable import PrettyTable

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class SystemsTable(object):

    def __init__(self, columns, isRank=False):
        self.columns = columns
        self.isRank = isRank
        columnNames = ['ID', 'Origin', 'Composition']
        if self.isRank:
            columnNames.insert(1, 'Rank')
        for column, columnName in self.columns:
            columnNames.append(columnName)

        self.table = PrettyTable(columnNames)


    def update(self, ID: int, system, fitness, rank=None):
        row = [ID, system['howCome'], fitness.getFitnessByID('simpleMoleculeUtility.composition', ID)]
        if self.isRank:
            row.insert(1, rank)
        for column, columnName in self.columns:
            value = fitness.getFitnessByID(column, ID)
            if isinstance(value, float):
                value = f'{value: 6.3f}'
            row.append(value)

        self.table.add_row(row)


class CrystalSystemRepresentation(object):
    def __init__(self, RES_FOLDER: str, numStages: int, columns, **kwargs):
        self.RES_FOLDER = RES_FOLDER
        self.numStages = numStages
        self.columns = columns

    def __call__(self, systems: dict, optimizer):
        fitness = optimizer.fitnessConvergence
        io_gatheredPOSCARS = io.StringIO('')
        io_gatheredPOSCARS_unrelaxed = io.StringIO('')
        table_Individuals = SystemsTable(self.columns)
        content_origin = ''
        content_enthalpies = ''
        for ID, system in sorted(systems.items()):
            try:
                write(io_gatheredPOSCARS_unrelaxed, system[0]['structure'].atoms, 'vasp', label=f'EA{ID}',
                      sort=True, direct=True, vasp5=True, long_format=False)
            except:
                pass
            content_origin += f"{ID} {system[0]['howCome']} {system[0]['parent']}\n"

            if len(system) > 1:
                content_enthalpies += ','.join([f"{sys['enthalpy']:6.3f}" for sys in system[1:]]) + '\n'

            if len(system) == self.numStages + 1:
                table_Individuals.update(ID, optimizer.target.pool.allSystems[ID], optimizer.fitness)
                try:
                    write(io_gatheredPOSCARS, system[self.numStages]['structure'].atoms, 'vasp', label=f'EA{ID}',
                          sort=True, direct=True, vasp5=True, long_format=False)
                except:
                    pass

        os.makedirs(self.RES_FOLDER, exist_ok=True)

        with open(os.path.join(self.RES_FOLDER, 'gatheredPOSCARS_unrelaxed'), 'w') as f:
            io_gatheredPOSCARS_unrelaxed.seek(0)
            shutil.copyfileobj(io_gatheredPOSCARS_unrelaxed, f)
        with open(os.path.join(self.RES_FOLDER, 'Individuals'), 'w') as f:
            f.write(table_Individuals.table.get_string() + '\n')
        with open(os.path.join(self.RES_FOLDER, 'gatheredPOSCARS'), 'w') as f:
            io_gatheredPOSCARS.seek(0)
            shutil.copyfileobj(io_gatheredPOSCARS, f)
        with open(os.path.join(self.RES_FOLDER, 'origin'), 'w') as f:
            f.write(content_origin)
        with open(os.path.join(self.RES_FOLDER, 'enthalpies_complete.csv'), 'w') as f:
            f.write(content_enthalpies)

    def drawESeries(self, systems, numStages):
        enths = []
        for system_stages in systems.values():
            if len(system_stages) == numStages + 1:
                enths.append([system['enthalpy'] for system in system_stages[1:]])
        enths = np.asarray(enths, dtype=float)
        plt.clf()
        Nplots = enths.shape[1] - 1
        Nrows = np.ceil(np.sqrt(Nplots)).astype(int)
        Ncolumns = np.ceil(Nplots/Nrows).astype(int)
        for i in range(Nplots):
            plt.subplot(Nrows, Ncolumns, i+1)
            plt.plot(enths[:, i], enths[:, i+1], 'go')
            plt.ylabel(f'E{i+2}')
            plt.xlabel(f'E{i+1}')
        plt.savefig(pj(self.RES_FOLDER, 'E_series.svg'))
