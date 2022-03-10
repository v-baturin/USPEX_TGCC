import os
import io
import shutil
import matplotlib
import numpy as np
from copy import copy
from collections import Counter
from collections.abc import Mapping
from itertools import combinations
from ase.atoms import Atoms
from ase.io.vasp import write_vasp, read_vasp
from os.path import join as pj
from prettytable import PrettyTable
import matplotlib.pyplot as plt

from .formatters import createHeader_wrap
from ..Presets import presetOutput

matplotlib.use('Agg')

MOL_CRYSTALS_PAPERS = '''\
Zhu Q., Oganov A.R., Glass C.W., Stokes H. (2012)
Constrained evolutionary algorithm for structure prediction of
molecular crystals: methodology and applications.
Acta Cryst. B, 68, 215-226 \
'''

VARCOMP_PAPERS = '''\
Lyakhov A.O., Oganov A.R., Valle M. (2010)
Crystal structure prediction using evolutionary approach.
In: Modern methods of crystal structure prediction (ed: A.R. Oganov)
Berlin: Wiley-VCH

Oganov A.R., Ma Y., Lyakhov A.O., Valle M., Gatti C. (2010)
Evolutionary crystal structure prediction as a method
for the discovery of minerals and materials.
Rev. Mineral. Geochem. 71, 271-298\
'''

EXTENDED_CONVEX_HULL_ENERGY_RANGE = 0.5


class SystemsTable(object):

    def __init__(self, columns, isRank=False):
        self.columns = columns
        self.isRank = isRank
        columnNames = ['ID', 'Origin']
        if self.isRank:
            columnNames.insert(1, 'Rank')
        for column, columnName in self.columns:
            columnNames.append(columnName)

        self.table = PrettyTable(columnNames)


    def update(self, ID: int, system, fitness, rank=None):
        row = [ID, system['howCome']]
        originalID = system['originalID'] if 'originalID' in system else ID
        if self.isRank:
            row.insert(1, rank)
        for column, columnName in self.columns:
            value = fitness.getFitnessByID(column, originalID)
            if value is None:
                value = fitness.getFitnessDirect(column, system)
            if isinstance(value, float):
                value = f'{value: 6.3f}'
            elif isinstance(value, Mapping):
                value = '  '.join(f'{key}: {amount}' for key, amount in value.items())
            row.append(value)

        self.table.add_row(row)


class AtomisticRepresentation(object):

    structureType = None
    atomType = None
    cellType = None
    atomicDisassemblerType = None

    def __init__(self, RES_FOLDER: str, columns, toDraw,
                 rangeECH = EXTENDED_CONVEX_HULL_ENERGY_RANGE, **kwargs):
        self.RES_FOLDER = RES_FOLDER
        self.columns = columns
        self.toDraw = toDraw
        self.rangeECH = rangeECH

    def getNewSystemsTable(self, isRank=False):
        return SystemsTable(self.columns, isRank)

    def presentSystems(self, systems: dict, optimizer, numStages):
        fitness = optimizer.optType
        io_gatheredPOSCARS = io.StringIO('')
        io_gatheredPOSCARS_unrelaxed = io.StringIO('')
        table_Individuals = self.getNewSystemsTable()
        content_origin = ''
        content_enthalpies = ''
        for ID, system in sorted(systems.items()):
            self.writeAtomicStructure(io_gatheredPOSCARS_unrelaxed, system[0])
            content_origin += f"{ID} {system[0]['howCome']} {system[0]['parent']}\n"

            if len(system) > 1:
                content_enthalpies += ','.join([f"{sys['enthalpy']:6.3f}" for sys in system[1:]]) + '\n'

            if len(system) == numStages + 1:
                table_Individuals.update(ID, optimizer.pool.allSystems[ID], optimizer.fitness)
                self.writeAtomicStructure(io_gatheredPOSCARS, system[numStages])

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
        self.drawESeries(systems, numStages)

    def drawESeries(self, systems, numStages):
        enths = []
        for system_stages in systems.values():
            if len(system_stages) == numStages + 1:
                enths.append([system['enthalpy'] for system in system_stages[1:]])
        if enths:
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
        coordinates = structure.getCartesianCoordinates()
        cell = structure.getCell().getEnvelopeCell(coordinates, 1)
        coordinates = cell.center(coordinates)
        atoms = Atoms([el.short_name for el in structure.getAtomTypes()], coordinates, cell=cell.getCellVectors())
        write_vasp(fileDescriptor, atoms, label=f"EA{system['ID']}", sort=True, direct=True, vasp5=True, long_format=False)

    @classmethod
    def readAtomicStructureRaw(cls, fileDescriptor, pbc=(1,1,1)) -> dict:
        atoms = read_vasp(fileDescriptor)
        atomTypes = [cls.atomType(s) for s in atoms.get_chemical_symbols()]
        cell = cls.cellType(atoms.get_cell().array, pbc)
        coordinates = atoms.get_positions()
        # cell = cls.cellType(atoms.get_cell().array, pbc).getEnvelopeCell(coordinates)
        # coordinates = cell.center(coordinates)
        return cls.structureType(atomTypes, coordinates, cell = cell)

    @classmethod
    def readAtomicStructure(cls, fileDescriptor, disassembler = None, pbc=(1,1,1)) -> dict:
        atoms = read_vasp(fileDescriptor)
        disassembler = cls.atomicDisassemblerType.createFlatDisassembler(len(atoms)) if disassembler is None else disassembler
        atomTypes = [cls.atomType(s) for s in atoms.get_chemical_symbols()]
        cell = cls.cellType(atoms.get_cell().array, pbc)
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

    @classmethod
    def getParametersBlock(cls, target) -> list:
        header = []
        ut = target.utilities
        isMolSystem = ut.simpleMoleculeUtility.isTrueMolecular
        isVarComp = not ut.compositionSpace.isFixedComposition
        dim = ut.cellUtility.getDim()
        hasEnv = ut.environmentUtility.hasEnvironment()


        # ---------------------------------------------------------------------------

        formatted_rows = createHeader_wrap(['Block for system description'], 'center')
        formatted_rows.append('')

        row = '    System type          :  Atomistic\n'
        row += f'    Dimension            :  {dim}\n'
        row += f'    Molecular            :  {"Yes" if isMolSystem else "No"}\n'
        row += f'    Variable composition :  {"Yes" if isVarComp else "No"}\n'
        row += f'    Has environment      :  {"Yes" if hasEnv else "No"}\n'


        formatted_rows.append(row)
        header += formatted_rows

        compositionSpace = ut.compositionSpace
        rows = ['    The investigated system is (block -- range): ']
        symbols = compositionSpace.symbols
        for block, rng in zip(compositionSpace.blocks, compositionSpace.range):
            rows.append(f'        {"".join(f"<{symbols[i]}>{block[i]}" for i in np.flatnonzero(block))}  --  {rng}')
        rows.append('')
        header += rows

        cell = ut.cellUtility.getCell()

        if cell is not None:
            lattice = cell.getCellVectors()
            rows = ['    This is a fixed lattice calculation ',
                    f'        {lattice[0, 0]:.4}   {lattice[0, 1]:.4}    {lattice[0, 2]:.4}',
                    f'        {lattice[1, 0]:.4}   {lattice[1, 1]:.4}    {lattice[1, 2]:.4}',
                    f'        {lattice[2, 0]:.4}   {lattice[2, 1]:.4}    {lattice[2, 2]:.4}']
        else:
            rows = ['    Volume (estimated) for blocks :']
            for block in compositionSpace.blocks:
                comp = Counter()
                for s, b in zip(symbols, block):
                    comp += ut.simpleMoleculeUtility.getElementalComposition({s:b})
                volume = ut.ionDistances.volumeEstimator.calcCompositionVolume(comp, ut.conditions.externalPressure)
                rows.append(f'        {"".join(f"<{symbols[i]}>{block[i]}" for i in np.flatnonzero(block))}  --  {volume:.4} A^3')

        rows.append('')
        header += rows

        # ---------------------------------------------------------------------------

        if isMolSystem:
            molecules = []
            molSymbols = []
            for symbol in symbols:
                molecule = ut.simpleMoleculeUtility.molecules[symbol]
                if len(molecule) > 1:
                    molecules.append(molecule)
                    molSymbols.append(symbol)
            rows = [f'    There is(are) {len(molecules)} type(s) of molecules in the system: ',
                  *(f'        <{symbol}> -- {mol.getFormula()}' for symbol, mol in zip(molSymbols, molecules)),
                     '    Please see the MOL_* files for the details.',
                     '']
            for symbol, molecule in zip(molSymbols, molecules):
                rows += [f'    The calculated Zmatrix for {symbol} is:',
                         cls.getZmatrixRepresentation(molecule, ut.simpleMoleculeUtility),
                         '']
            header += rows

        # ---------------------------------------------------------------------------

        if isMolSystem:
            header += createHeader_wrap(['Molecular Crystals suggested papers:'], 'center')
            header += createHeader_wrap([text.rstrip() for text in MOL_CRYSTALS_PAPERS.split('\n')], 'left')

        if isVarComp:
            header += createHeader_wrap(['Variable Composition suggested papers:'], 'center')
            header += createHeader_wrap([text.rstrip() for text in VARCOMP_PAPERS.split('\n')], 'left')

        # ---------------------------------------------------------------------------

        text = ['Block for atomic description']
        formatted_rows = createHeader_wrap(text, 'center')
        formatted_rows.append('')

        symbols = set()
        for symbol in ut.compositionSpace.symbols:
            symbols.update(ut.simpleMoleculeUtility.molecules[symbol].getAtomTypes())
        symbols = sorted(symbols)
        minDistMatrix = ut.ionDistances.getDistances(symbols, ut.conditions.externalPressure)

        row = '    There are %1d types of atoms in the system:' % len(symbols)
        for symbol in symbols:
            row += '%5s' % symbol
        row += '\n'

        for i, symbol in enumerate(symbols):
            row += '    Minimum distances:                 %5s: ' % symbol
            for j in range(len(symbols)):
                row += '%4.2f  ' % minDistMatrix[i, j]
            row += '\n'
        row += '\n'

        for symbol1 in symbols:
            row += '           Good Bonds:                 %5s: ' % symbol1
            for symbol2 in symbols:
                row += '%4.2f  ' % (symbol1.good_bonds*symbol2.good_bonds) ** 0.5
            row += '\n'
        row += '\n'

        row += '             Valences:                        '
        for symbol in symbols:
            row += '%4.2f  ' % symbol.valence
        row += '\n'

        formatted_rows.append(row)
        header += formatted_rows

        # ---------------------------------------------------------------------------

        formatted_rows = createHeader_wrap(['Conditions'], 'center')
        formatted_rows.append('')
        if ut.conditions.externalPressure > 0:
            formatted_rows.append('* External Pressure is: %6.4f GPa *' % ut.conditions.externalPressure)
        formatted_rows.append('')

        header += formatted_rows

        return header


    @staticmethod
    def getPopulationSummaryBlock(population, optimizer) -> list:
        utlts = optimizer.target.utilities
        if utlts.cellUtility.getDim() == 3:
            numBlocks = [utlts.compositionSpace.numBlocks(utlts.simpleMoleculeUtility.composition(system)) for system in population]
            numBlocks = np.asarray(numBlocks)
            volumes = [optimizer.fitness.getFitnessDirect('cellUtility.volume', system) for system in population]
            volumes = np.asarray(volumes)
            approximateVolume = ' '.join(f'{float(vol):.4} A^3' for vol in np.linalg.lstsq(numBlocks, volumes)[0])
        else:
            approximateVolume = 'NA'
        originalID = lambda system: system['originalID'] if 'originalID' in system else system['ID']
        fitness = [optimizer.fitness.getFitnessByID(optimizer.optType, originalID(system)) for system in population if not system['isBad']]
        order = [optimizer.target.utilities.radialDistributionUtility.averageOrder(system) for system in population if not system['isBad']]
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
        qe /= -len(comb) if comb else 1

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

    def presentOptimizer(self, optimizers, optimizer):
        originalID = lambda system: system['originalID'] if 'originalID' in system else system['ID']
        content_BESTIndividuals = ''
        content_convexHull = ''
        table_goodStructures = self.getNewSystemsTable(isRank=True)
        table_extendedConvexHull = self.getNewSystemsTable( isRank=True)
        io_BESTgatheredPOSCARS = io.StringIO('')
        io_goodStructuresPOSCARS = io.StringIO('')
        io_extendedConvexHullPOSCARS = io.StringIO('')

        os.makedirs(self.RES_FOLDER, exist_ok=True)

        fitness = optimizer.optType

        for generation, opt in enumerate(optimizers):
            content_BESTIndividuals += f'Generation {generation}\n'
            pool = opt.pool
            table = self.getNewSystemsTable()
            for ID in opt.best:
                table.update(ID, pool.allSystems[ID], opt.fitness)
            content_BESTIndividuals += table.table.get_string() + '\n'
        with open(pj(self.RES_FOLDER, 'BESTIndividuals'), 'w') as fp:
            fp.write(content_BESTIndividuals)

        for opt in optimizers:
            pool = opt.pool
            for ID in opt.best:
                AtomisticRepresentation.writeAtomicStructure(io_BESTgatheredPOSCARS, pool.allSystems[ID])
        with open(pj(self.RES_FOLDER, 'BESTgatheredPOSCARS'), 'w') as fp:
            io_BESTgatheredPOSCARS.seek(0)
            shutil.copyfileobj(io_BESTgatheredPOSCARS, fp)

        if len(optimizer.target.utilities.compositionSpace.blocks) == 1:
            allFitnesses = optimizer.fitness.getAllFitnesses(fitness)
            fronts = optimizer.fitness.sort(list(optimizer.pool.uniqueSystems), allFitnesses)
            for rank, front in enumerate(fronts):
                for system in front:
                    table_goodStructures.update(system['ID'], system, optimizer.fitness, rank=rank)
                    AtomisticRepresentation.writeAtomicStructure(io_goodStructuresPOSCARS, system)

            with open(pj(self.RES_FOLDER, 'goodStructures'), 'w') as fp:
                fp.write(table_goodStructures.table.get_string() + '\n')

            with open(pj(self.RES_FOLDER, 'goodStructures_POSCARS'), 'w') as fp:
                io_goodStructuresPOSCARS.seek(0)
                shutil.copyfileobj(io_goodStructuresPOSCARS, fp)


        else:
            for generation, opt in enumerate(optimizers):
                convexHull = [system for system in opt.pool.uniqueSystems
                              if np.isclose(opt.fitness.getFitnessByID('enthalpyCCH', originalID(system)), 0.0)]
                content_convexHull += f'Generation {generation}\n'
                table = self.getNewSystemsTable()
                for system in convexHull:
                    table.update(system['ID'], system, opt.fitness)
                content_convexHull += table.table.get_string() + '\n'

            with open(pj(self.RES_FOLDER, 'convex_hull'), 'w') as fp:
                fp.write(content_convexHull)

            extendedConvexHull = [system for system in optimizer.pool.uniqueSystems
                                  if optimizer.fitness.getFitnessByID('enthalpyCCH', originalID(system)) < self.rangeECH]

            allFitnesses = {system['ID'] : optimizer.pool.generations[-1]['fitness'].getFitnessByID(fitness, originalID(system))
                            for system in extendedConvexHull}
            fronts = optimizer.fitness.sort(extendedConvexHull, allFitnesses)

            for rank, front in enumerate(fronts):
                for system in front:
                    table_extendedConvexHull.update(system['ID'], system, optimizer.fitness, rank=rank)
            with open(pj(self.RES_FOLDER, 'extended_convex_hull'), 'w') as fp:
                fp.write(table_extendedConvexHull.table.get_string())

            for front in fronts:
                for system in front:
                    AtomisticRepresentation.writeAtomicStructure(io_extendedConvexHullPOSCARS, system)
            with open(pj(self.RES_FOLDER, 'extended_convex_hull_POSCARS'), 'w') as fp:
                io_extendedConvexHullPOSCARS.seek(0)
                shutil.copyfileobj(io_extendedConvexHullPOSCARS, fp)

            compositionSpace = optimizer.target.utilities.compositionSpace
            if len(compositionSpace.blocks) == 2 and convexHull:
                self._drawExtendedConvexHull2(compositionSpace, convexHull, extendedConvexHull)
            elif len(compositionSpace.blocks) == 3 and convexHull:
                self._drawExtendedConvexHull3(compositionSpace, convexHull, extendedConvexHull)

        self._drawProperties(optimizer.pool.uniqueSystems, optimizer.fitness)


    def _drawProperties(self, uniqueSystems, fitness):
        originalID = lambda system: system['originalID'] if 'originalID' in system else system['ID']
        for type, propertyY, typeY, propertyX, typeX in self.toDraw:
            if type == 'dep':
                Y = []
                X = []
                for system in uniqueSystems:
                    valueX = fitness.getFitnessByID(propertyX, originalID(system))
                    valueY = fitness.getFitnessByID(propertyY, originalID(system))
                    if typeY == 'raw':
                        Y.append(valueY)
                    elif typeY == 'per_atom':
                        Y.append(valueY/len(system['molecules']))
                    if typeX == 'raw':
                        X.append(valueX)
                    elif typeX == 'per_atom':
                        X.append(valueX/len(system['molecules']))
                plt.clf()
                plt.plot(X,Y,'go')
                plt.ylabel(f'{propertyY}({typeY})')
                plt.xlabel(f'{propertyX}({typeX})')
                plt.savefig(pj(self.RES_FOLDER, f'{propertyY}({typeY})_vs_{propertyX}({typeX}).svg'))
            elif type == 'stat':
                Y = []
                for system in uniqueSystems:
                    value = fitness.getFitnessByID(propertyY, originalID(system))
                    if not np.isinf(value):
                        if typeY == 'raw':
                            Y.append(value)
                        elif typeY == 'per_atom':
                            Y.append(value/len(system['molecules']))
                plt.clf()
                plt.hist(Y, len(Y)//10+1, facecolor='g', alpha=0.75)
                plt.savefig(pj(self.RES_FOLDER, f'{propertyY}({typeY})_statistics.svg'))

    def _drawExtendedConvexHull2(self, compositionSpace, convexHull, extendedConvexHull):
        leftNumBlocks = np.asarray(compositionSpace.numBlocks(convexHull[0]['simpleMoleculeUtility.composition']), dtype = float)
        leftNumBlocksTotal = np.sum(leftNumBlocks)
        leftNumBlocks /= leftNumBlocksTotal
        leftEnthalpy = convexHull[0]['enthalpy']/leftNumBlocksTotal
        rightNumBlocks = leftNumBlocks
        rightEnthalpy = leftEnthalpy
        for system in convexHull:
            numBlocks = np.asarray(compositionSpace.numBlocks(system['simpleMoleculeUtility.composition']), dtype = float)
            numBlocksTotal = np.sum(numBlocks)
            numBlocks /= numBlocksTotal
            if numBlocks[1] < leftNumBlocks[1]:
                leftNumBlocks = numBlocks
                leftEnthalpy = system['enthalpy'] / numBlocksTotal
            elif numBlocks[1] > rightNumBlocks[1]:
                rightNumBlocks = numBlocks
                rightEnthalpy = system['enthalpy'] / numBlocksTotal
        Xch = []
        Ych = []
        for system in convexHull:
            numBlocks = np.asarray(compositionSpace.numBlocks(system['simpleMoleculeUtility.composition']), dtype = float)
            numBlocksTotal = np.sum(numBlocks)
            numBlocks /= numBlocksTotal
            C = np.array([leftNumBlocks, rightNumBlocks])
            E = np.array([leftEnthalpy, rightEnthalpy])
            Enthalpy = system['enthalpy']/numBlocksTotal - np.dot(np.linalg.lstsq(C.T, numBlocks)[0], E)
            Xch.append(numBlocks[1])
            Ych.append(Enthalpy)
        inds = np.argsort(Xch)
        Xch = np.asarray(Xch)[inds]
        Ych = np.asarray(Ych)[inds]
        X = []
        Y = []
        for system in extendedConvexHull:
            numBlocks = np.asarray(compositionSpace.numBlocks(system['simpleMoleculeUtility.composition']), dtype = float)
            numBlocksTotal = np.sum(numBlocks)
            numBlocks /= numBlocksTotal
            C = np.array([leftNumBlocks, rightNumBlocks])
            E = np.array([leftEnthalpy, rightEnthalpy])
            Enthalpy = system['enthalpy']/numBlocksTotal - np.dot(np.linalg.lstsq(C.T, numBlocks)[0], E)
            X.append(numBlocks[1])
            Y.append(Enthalpy)
        np.savetxt(pj(self.RES_FOLDER, 'ExtendedConvexHull.csv'), np.stack((X,Y), axis=-1), fmt='%6.3f', delimiter=',')
        plt.clf()
        plt.plot(X,Y,'go')
        plt.plot(Xch,Ych,'b-o')
        plt.ylabel('Enthalpy of formation (eV/atom)')
        components = []
        for block in compositionSpace.blocks:
            components.append(''.join(f'{symbol}{mult}' for symbol,mult in zip(compositionSpace.symbols, block)))
        plt.xlabel(f'Composition ratio: {components[1]}/({components[0]}+{components[1]})')
        plt.savefig(pj(self.RES_FOLDER, 'ExtendedConvexHull.svg'))


    def _drawExtendedConvexHull3(self, compositionSpace, convexHull, extendedConvexHull):
        pass

    @staticmethod
    def applyPresetOutputParameters(optimizer, output):
        dim = optimizer.target.utilities.cellUtility.getDim()
        if dim == 3:
            prefix = 'Crystal'
        elif dim == 2:
            prefix = 'Nano2D'
        elif dim == 1:
            prefix = 'Nano1D'
        elif dim == 0:
            prefix = 'Nano0D'
        else:
            raise RuntimeError(f'Wrong dim {dim}.')
        if optimizer.target.utilities.compositionSpace.isFixedComposition:
            suffix = 'FixComp'
        else:
            suffix = 'VarComp'
        default = copy(presetOutput[prefix + suffix])
        default.update(output)
        return default
