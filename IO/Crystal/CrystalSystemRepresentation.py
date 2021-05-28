import os
import io
import shutil
import matplotlib
import numpy as np

from collections import Counter
from itertools import combinations
from ase.atoms import Atoms
from ase.io.vasp import write_vasp, read_vasp
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

    structureType = None
    atomType = None
    cellType = None
    atomicDisassemblerType = None

    def __init__(self, RES_FOLDER: str, numStages: int, columns, **kwargs):
        self.RES_FOLDER = RES_FOLDER
        self.numStages = numStages
        self.columns = columns

    def __call__(self, systems: dict, optimizer):
        fitness = optimizer.optType
        io_gatheredPOSCARS = io.StringIO('')
        io_gatheredPOSCARS_unrelaxed = io.StringIO('')
        table_Individuals = SystemsTable(self.columns)
        content_origin = ''
        content_enthalpies = ''
        for ID, system in sorted(systems.items()):
            self.writeAtomicStructure(io_gatheredPOSCARS_unrelaxed, system[0])
            content_origin += f"{ID} {system[0]['howCome']} {system[0]['parent']}\n"

            if len(system) > 1:
                content_enthalpies += ','.join([f"{sys['enthalpy']:6.3f}" for sys in system[1:]]) + '\n'

            if len(system) == self.numStages + 1:
                table_Individuals.update(ID, optimizer.target.pool.allSystems[ID], optimizer.fitness)
                self.writeAtomicStructure(io_gatheredPOSCARS, system[self.numStages])

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


    @classmethod
    def writeAtomicStructure(cls, fileDescriptor, system: dict):
        structure, disassembler = cls.structureType.assemble(**system)
        atoms = Atoms([el.short_name for el in structure.getAtomTypes()], structure.getCartesianCoordinates(),
                      cell = structure.getCell().getCellVectors())
        write_vasp(fileDescriptor, atoms, label=f"EA{system['ID']}", sort=True, direct=True, vasp5=True, long_format=False)

    @classmethod
    def readAtomicStructure(cls, fileDescriptor, disassembler = None) -> dict:
        atoms = read_vasp(fileDescriptor)
        disassembler = cls.atomicDisassemblerType.createFlatDisassembler(len(atoms)) if disassembler is None else disassembler
        atomTypes = [cls.atomType(s) for s in atoms.get_chemical_symbols()]
        cell = cls.cellType(atoms.get_cell().array, tuple(atoms.get_pbc()))
        return disassembler.disassemble(cls.structureType(atomTypes, atoms.get_positions(), cell = cell))

    @classmethod
    def getZmatrixRepresentation(cls, molecule, utility) -> str:
        elements = molecule.getAtomTypes()
        coordinates = molecule.getCartesianCoordinates()
        zmatrixConfig = molecule.getZmatrixConfig()
        zmatrix = utility.coordToZmatrix(coordinates, zmatrixConfig)
        repr = ['Atom Bond-length Bond-angle Torsion-angle   i   j   k',
                '      (Angstrom)  (Degree)    (Degree)',
             *(f'{el.short_name:2}    {zrow[0]:8.4}    {zrow[1]*180/np.pi:8.4}    {zrow[2]*180/np.pi:8.4}    {fmt[0]:3} {fmt[1]:3} {fmt[2]:3}'
               for el, zrow, fmt in zip(elements, zmatrix, zmatrixConfig))]
        return '\n'.join(repr)

    @classmethod
    def registerTypes(cls, structureType, atomType, cellType, atomicDisassemblerType):
        cls.structureType = structureType
        cls.atomType = atomType
        cls.cellType = cellType
        cls.atomicDisassemblerType = atomicDisassemblerType


def getPopulationSummaryBlock(population, optimizer) -> list:
    utlts = optimizer.target.utilities
    numBlocks = [utlts.compositionSpace.numBlocks(utlts.simpleMoleculeUtility.composition(system)) for system in population]
    numBlocks = np.asarray(numBlocks)
    volumes = [optimizer.fitness.getFitnessByID('cellUtility.volume', system['ID'])for system in population]
    volumes = np.asarray(volumes)
    approximateVolume = ' '.join(f'{float(vol):.4} A^3' for vol in np.linalg.lstsq(numBlocks, volumes)[0])
    fitness = [optimizer.fitness.getFitnessByID(optimizer.optType, system['ID']) for system in population]
    order = [optimizer.target.utilities.radialDistributionUtility.averageOrder(system) for system in population]
    if np.any(np.isnan(np.asarray(fitness, dtype = float))):
        correlation = 0.0
    else:
        correlation = np.corrcoef(order, fitness)[0, 1]

    qe = 0
    comb = list(combinations(population, 2))
    for s1, s2 in comb:
        tmp_fing1 = optimizer.target.utilities.radialDistributionUtility.structureFingerprint(s1)
        tmp_fing2 = optimizer.target.utilities.radialDistributionUtility.structureFingerprint(s2)
        dist = tmp_fing1.cosine_distance(tmp_fing1, tmp_fing2)
        qe += (1 - dist) * np.log(1 - dist)
    qe /= -len(comb)

    block = [ '    Generation Summary',
             f'      Correlation coefficient: {correlation:.4}',
             f'      Approximate volume(s)  : {approximateVolume}',
             f'      Quasi entropy          : {qe:.4}']

    if not utlts.compositionSpace.isFixedComposition:
        numIons = [utlts.compositionSpace.numIons(utlts.simpleMoleculeUtility.composition(system)) for system in population]
        numIons = np.asarray(numIons)
        comps = numIons/np.sum(numIons, axis=1).reshape((-1,1))
        combs = list(combinations(comps, 2))
        compositionEntropy = 0
        for c1, c2 in combs:
            cos_dist = (np.dot(c1, c2)) / (np.linalg.norm(c1) * np.linalg.norm(c2))
            if abs(cos_dist) < 0.001:
              compositionEntropy -= cos_dist
            else:
              compositionEntropy += cos_dist*np.log(cos_dist)
        compositionEntropy /= -len(combs)
        compositionCounter = Counter()
        for c in comps:
            cStr = ' '.join(f'{n:.3}' for n in c)
            compositionCounter[cStr] += 1

        block.append(f'      Composition entropy    : {compositionEntropy:.4}')
        block.append(f'      Number of compositions : {len(compositionCounter)}')

    return block