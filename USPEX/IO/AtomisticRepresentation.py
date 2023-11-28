import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from collections.abc import Mapping
from itertools import combinations, chain
from pathlib import Path
from prettytable import PrettyTable
from itertools import zip_longest

from .formatters import createHeader_wrap
from ..Expressions.Functions.presets import presetFitness, applyPresetsRecursive

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


presetLabels = {
    '.enthalpy': 'Enthalpy (eV)',
    '.energy': 'Energy (eV)',
    # 'enthalpyCCH': 'Enthalpy above CH (eV/block)',
    # 'enthalpyCS': 'Enthalpy above the best for composition(eV/block)',
    'simpleMoleculeUtility.composition': 'Composition',
    'simpleMoleculeUtility.density': 'Density (g/cm^3)',
    'cellUtility.volume': 'Volume (A^3)',
    'cellUtility.area': 'Area (A^2)',
    'cellUtility.length': 'Period (A)',
    'cellUtility.symmetry': 'SYMMETRY (N)',
    'bondUtility.hardness': 'Hardness',
    'radialDistributionUtility.structureOrder': 'Structure order',
    'radialDistributionUtility.averageOrder': 'Average order',
    'radialDistributionUtility.quasientropy': 'Quasientropy',
    'elasticML.youngsModulus': 'ML Youngs Modulus (Gpa)',
    'elasticML.bulkModulus': 'ML Bulk Modulus (GPa)',
    'elasticML.shearModulus': 'ML Shear Modulus (GPa)',
    'elasticML.poissonsRatio': 'ML Poissons Ratio',
    'elasticML.pughsRatio': 'ML Pughs Ratio',
    'elasticML.vickersHardness': 'ML Vickers Hardness (GPa)',
    'elasticML.fractureToughness': 'ML Fracture Toughness (MPa*m^1/2)'
}

def getPresetLables(expression):
    if isinstance(expression, str):
        ext, prop, suffix = expression.split('.')
        if f'{ext}.{prop}' in presetLabels:
            return presetLabels[f'{ext}.{prop}']
    return ''


class SystemsTable(object):

    def __init__(self, columns, pool, isRank=False):
        self.columns = [pool.createExpression(column) for column in columns]
        self.isRank = isRank
        columnNames = ['ID', 'Origin']
        if self.isRank:
            columnNames.insert(1, 'Rank')
        for column in self.columns:
            columnNames.append(getPresetLables(column))

        self.table = PrettyTable(columnNames)


    def update(self, ID: int, system, rank=None):
        row = [ID, system['.howCome.origin']]
        if self.isRank:
            row.insert(1, rank)
        for column in self.columns:
            try:
                value = system[column]
            except Exception:
                value = None
            if isinstance(value, float):
                value = f'{value: 6.3f}'
            elif isinstance(value, Mapping):
                value = '  '.join(f'{key}: {amount}' for key, amount in value.items())
            row.append(value)

        self.table.add_row(row)


class AtomisticRepresentation(object):

    Atomistic = None

    @classmethod
    def registerTypes(cls, Atomistic):
        cls.Atomistic = Atomistic

    def __init__(self, RES_FOLDER: str, columns, stages, toDraw, presentConvexHull=None, presentPareto=(),
                 rangeECH = EXTENDED_CONVEX_HULL_ENERGY_RANGE, **kwargs):
        self.RES_FOLDER = Path(RES_FOLDER)
        self.columns = [applyPresetsRecursive(column) for column in columns]
        self.stages = stages
        self.toDraw = toDraw
        self.presentConvexHull = applyPresetsRecursive(presentConvexHull)
        self.presentPareto = [applyPresetsRecursive(expr) for expr in presentPareto]
        self.rangeECH = rangeECH

    def getNewSystemsTable(self, pool, isRank=False):
        return SystemsTable(self.columns, pool, isRank)

    def presentSystems(self, optimizer):
        systems = optimizer.allSystems
        systems_gatheredPOSCARS = []
        systems_gatheredPOSCARS_unrelaxed = []
        if optimizer.generations:
            table_Individuals = self.getNewSystemsTable(optimizer.generations[-1].goodSystems)
        else:
            table_Individuals = self.getNewSystemsTable(optimizer.allSystems)
        content_origin = ''
        content_enthalpies = ''
        for ID in systems.getIDs():
            system = systems.getEntry(ID)
            unrelaxed = system.getFlavour('origin')
            unrelaxed.setProperty('label', f"EA{ID}")
            systems_gatheredPOSCARS_unrelaxed.append(unrelaxed)
            content_origin += f"{ID} {unrelaxed['.howCome']} {unrelaxed['.parent']}\n"

            content_enthalpies += ','.join([f"{system[f'.enthalpy.{stage}']:6.3f}"
                                            for stage in self.stages if f'.enthalpy.{stage}' in system]) + '\n'

            table_Individuals.update(ID, system)

            if str(optimizer.target.defaultSuffix) in system.flavours:
                final = system.getFlavour(str(optimizer.target.defaultSuffix))
                final.setProperty('label', f"EA{system.ID}")
                systems_gatheredPOSCARS.append(final)

        self.RES_FOLDER.mkdir(parents=True, exist_ok=True)

        self.Atomistic.writeAtomicStructures(self.RES_FOLDER/'gatheredPOSCARS_unrelaxed',
                                   systems_gatheredPOSCARS_unrelaxed)
        self.Atomistic.writeAtomicStructures(self.RES_FOLDER/'gatheredPOSCARS',
                                   systems_gatheredPOSCARS)
        with open(self.RES_FOLDER/'Individuals', 'w') as f:
            f.write(table_Individuals.table.get_string() + '\n')
        with open(self.RES_FOLDER/'origin', 'w') as f:
            f.write(content_origin)
        with open(self.RES_FOLDER/'enthalpies_complete.csv', 'w') as f:
            f.write(content_enthalpies)
        self.drawESeries(systems)

    def drawESeries(self, systems):
        enths = []
        for ID in systems.getIDs():
            system = systems.getEntry(ID)
            try:
                enths.append([system[f'.enthalpy.{suffix}'] for suffix in self.stages])
            except Exception:
                pass
        if enths:
            enths = np.asarray(enths, dtype=float)
            plt.figure()
            Nplots = enths.shape[1] - 1
            Nrows = np.ceil(np.sqrt(Nplots)).astype(int)
            Ncolumns = np.ceil(Nplots/Nrows).astype(int)
            for i in range(Nplots):
                plt.subplot(Nrows, Ncolumns, i+1)
                plt.plot(enths[:, i], enths[:, i+1], 'go')
                plt.ylabel(f'E{i+2}')
                plt.xlabel(f'E{i+1}')
            plt.savefig(self.RES_FOLDER/'E_series.svg')
            plt.close()

    @classmethod
    def getZmatrixRepresentation(cls, molecule, utility) -> str:
        elements = molecule.getAtomTypes()
        coordinates = molecule.getCartesianCoordinates()
        zmatrixConfig = molecule.getZmatrixConfig()
        if zmatrixConfig is not None:
            zmatrix = utility.coordToZmatrix(coordinates, zmatrixConfig)
            repr = ['Atom Bond-length Bond-angle Torsion-angle   i   j   k',
                    '      (Angstrom)  (Degree)    (Degree)',
                 *(f'{el.short_name:2}    {zrow[0]:8.4}    {zrow[1]*180/np.pi:8.4}    {zrow[2]*180/np.pi:8.4}    {fmt[0]:3} {fmt[1]:3} {fmt[2]:3}'
                   for el, zrow, fmt in zip(elements, zmatrix, zmatrixConfig))]
            return '\n'.join(repr)
        else:
            return 'No corresponding Z-matrix\n'

    @classmethod
    def getParametersBlock(cls, target) -> list:
        header = []
        ut = target.utilities
        isMolSystem = ut.simpleMoleculeUtility.isTrueMolecular
        isVarComp = not ut.compositionSpace.isFixedComposition
        dim = ut.cellUtility.getDim()
        hasEnv = len(ut.environmentUtility.environments) > 0
        hasJunct = ut.junctionUtility.hasJunctions


        # ---------------------------------------------------------------------------

        formatted_rows = createHeader_wrap(['Block for system description'], 'center')
        formatted_rows.append('')

        row = '    System type          :  Atomistic\n'
        row += f'    Dimension            :  {dim}\n'
        row += f'    Molecular            :  {"Yes" if isMolSystem else "No"}\n'
        row += f'    Variable composition :  {"Yes" if isVarComp else "No"}\n'
        row += f'    Has environment      :  {"Yes" if hasEnv else "No"}\n'
        row += f'    Has Junctions        :  {"Yes" if hasJunct else "No"}\n'


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

                volume = ut.bondUtility.volumeEstimator.calcCompositionVolume(comp, ut.conditions.externalPressure)
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
            # for symbol, molecule in zip(molSymbols, molecules):
            #     rows += [f'    The calculated Zmatrix for {symbol} is:',
            #              cls.getZmatrixRepresentation(molecule, ut.simpleMoleculeUtility),
            #              '']
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
        minDistMatrix = ut.bondUtility.getDistances(symbols, ut.conditions.externalPressure)

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


    def getPopulationSummaryBlock(self, population, optimizer) -> list:
        utlts = optimizer.target.utilities
        population = [population.getEntry(ID) for ID in population.getIDs()]
        if utlts.cellUtility.getDim() == 3:
            numBlocks = [utlts.compositionSpace.numBlocks(system['simpleMoleculeUtility.composition.origin']) for system in population]
            numBlocks = np.asarray(numBlocks)
            volumes = [system[f'cellUtility.volume.{optimizer.target.defaultSuffix}'] for system in population]
            volumes = np.asarray(volumes)
            approximateVolume = ' '.join(f'{float(vol):.4} A^3' for vol in np.linalg.lstsq(numBlocks, volumes)[0])
        else:
            approximateVolume = 'NA'
        # originalID = lambda system: system['originalID'] if 'originalID' in system else system['ID']
        if isinstance(optimizer.optType, str):
            optType = optimizer.optType
        else:
            optType = optimizer.generations[-1].goodSystems.createExpression(optimizer.optType)
        fitness = [system[optType] for system in population]
        order = [system[f'radialDistributionUtility.averageOrder.{optimizer.target.defaultSuffix}'] for system in population]
        if np.any(np.isnan(np.asarray(fitness, dtype=float))):
            correlation = 0.0
        else:
            correlation = np.corrcoef(order, fitness)[0, 1]

        qe = 0
        comb = list(combinations(population, 2))
        for s1, s2 in comb:
            # if not s1['isBad'] and not s2['isBad']:
            tmp_fing1 = s1[f'radialDistributionUtility.structureFingerprint.{optimizer.fingerprintUtility.suffix}']
            tmp_fing2 = s2[f'radialDistributionUtility.structureFingerprint.{optimizer.fingerprintUtility.suffix}']
            dist = tmp_fing1.cosine_distance(tmp_fing1, tmp_fing2)
            qe += (1 - dist) * np.log(1 - dist)
        qe /= -len(comb) if comb else 1

        block = [ '    Generation Summary',
                 f'      Correlation coefficient: {correlation:.4}',
                 f'      Approximate volume(s)  : {approximateVolume}',
                 f'      Quasi entropy          : {qe:.4}']

        if not utlts.compositionSpace.isFixedComposition:
            numIons = [utlts.compositionSpace.numIons(system['simpleMoleculeUtility.composition.origin']) for system in population]
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

    def presentOptimizer(self, optimizer):
        # originalID = lambda system: system['originalID'] if 'originalID' in system else system['ID']
        if not optimizer.generations:
            return
        content_BESTIndividuals = ''
        content_convexHull = ''
        table_goodStructures = self.getNewSystemsTable(optimizer.generations[-1].goodSystems, isRank=True)
        table_extendedConvexHull = self.getNewSystemsTable(optimizer.generations[-1].goodSystems, isRank=True)
        systems__BESTgatheredPOSCARS = []
        systems_goodStructuresPOSCARS = []
        systems_extendedConvexHullPOSCARS = []

        self.RES_FOLDER.mkdir(parents=True, exist_ok=True)

        for i, best in enumerate(optimizer.bestHistory):
            content_BESTIndividuals += f'Generation {i}\n'
            table = self.getNewSystemsTable(optimizer.generations[i].goodSystems)
            for ID in best:
                table.update(ID, optimizer.allSystems.getEntry(ID))
            content_BESTIndividuals += table.table.get_string() + '\n'
        with open(self.RES_FOLDER/'BESTIndividuals', 'w') as fp:
            fp.write(content_BESTIndividuals)

        for best in optimizer.bestHistory:
            for ID in best:
                system = optimizer.allSystems.getEntry(ID).getFlavour(str(optimizer.target.defaultSuffix))
                system.setProperty('label', f"EA{ID}")
                systems__BESTgatheredPOSCARS.append(system)
        self.Atomistic.writeAtomicStructures(self.RES_FOLDER/'BESTgatheredPOSCARS', systems__BESTgatheredPOSCARS)

        compositionSpace = optimizer.target.utilities.compositionSpace
        csSize = len(compositionSpace.blocks)

        if optimizer.generations:
            if isinstance(optimizer.optType, str):
                optType = optimizer.optType
            else:
                optType = optimizer.generations[-1].goodSystems.createExpression(optimizer.optType)
            fronts = optimizer.generations[-1].uniqueSystems.fronts(optType)
            if csSize == 1:
                for rank, front in enumerate(fronts):
                    for system in front:
                        ID = system.ID
                        table_goodStructures.update(ID, system, rank=rank)
                        s = system.getFlavour(str(optimizer.target.defaultSuffix))
                        s.setProperty('label', f"EA{ID}")
                        systems_goodStructuresPOSCARS.append(s)
                with open(self.RES_FOLDER/'goodStructures', 'w') as fp:
                    fp.write(table_goodStructures.table.get_string() + '\n')

                self.Atomistic.writeAtomicStructures(self.RES_FOLDER/'goodStructures_POSCARS', systems_goodStructuresPOSCARS)
            else:
                goodStructresFolder = self.RES_FOLDER/'goodStructures'
                goodStructresFolder.mkdir(parents=True, exist_ok=True)
                goodStructures = {}
                goodStructuresPOSCARS = {}
                for rank, front in enumerate(fronts):
                    for system in front:
                        numBlocks = tuple(compositionSpace.numBlocks(system['simpleMoleculeUtility.composition.origin']))
                        if numBlocks not in goodStructures:
                            goodStructures[numBlocks] = self.getNewSystemsTable(optimizer.generations[-1].goodSystems,
                                                                                isRank=True)
                            goodStructuresPOSCARS[numBlocks] = []
                        ID = system.ID
                        goodStructures[numBlocks].update(ID, system, rank=rank)
                        s = system.getFlavour(str(optimizer.target.defaultSuffix))
                        s.setProperty('label', f"EA{ID}")
                        goodStructuresPOSCARS[numBlocks].append(s)

                for comp, table_gs in goodStructures.items():
                    with open(goodStructresFolder/f'{"_".join(str(x) for x in comp)}', 'w') as fp:
                        fp.write(table_gs.table.get_string() + '\n')

                for comp, systems_gs_POSCARS in goodStructuresPOSCARS.items():
                    self.Atomistic.writeAtomicStructures(goodStructresFolder/f'{"_".join(str(x) for x in comp)}_POSCARS',
                                               systems_gs_POSCARS)

            self._drawProperties(optimizer.generations[-1].uniqueSystems)

            if self.presentConvexHull is not None:
                convexHull = []
                for i, generation in enumerate(optimizer.generations):
                    expr = generation.goodSystems.createExpression(self.presentConvexHull)
                    convexHull = []
                    for ID in generation.goodSystems.getIDs():
                        system = generation.goodSystems.getEntry(ID)
                        try:
                            if np.isclose(system.getExpression(expr), 0.0):
                                convexHull.append(system)
                        except Exception:
                            pass
                    content_convexHull += f'Generation {i}\n'
                    table = self.getNewSystemsTable(optimizer.generations[i].goodSystems)
                    for system in convexHull:
                        table.update(system.ID, system)
                    content_convexHull += table.table.get_string() + '\n'

                with open(self.RES_FOLDER/'convex_hull', 'w') as fp:
                    fp.write(content_convexHull)

                for rank, front in enumerate(fronts):
                    for system in front:
                        table_extendedConvexHull.update(system.ID, system, rank=rank)
                with open(self.RES_FOLDER/'extended_convex_hull', 'w') as fp:
                    fp.write(table_extendedConvexHull.table.get_string())

                for front in fronts:
                    for system in front:
                        ID = system.ID
                        system = system.getFlavour(str(optimizer.target.defaultSuffix))
                        system.setProperty('ID', ID)
                        systems_extendedConvexHullPOSCARS.append(system)
                self.Atomistic.writeAtomicStructures(self.RES_FOLDER/'extended_convex_hull_POSCARS',
                                           systems_extendedConvexHullPOSCARS)

                if csSize == 2:
                    self._drawExtendedConvexHull2(compositionSpace, convexHull + optimizer.extraData,
                                                  optimizer.generations[-1].uniqueSystems, optimizer.target.defaultSuffix)
                elif csSize == 3:
                    self._drawExtendedConvexHull3(compositionSpace, convexHull + optimizer.extraData,
                                                  optimizer.generations[-1].uniqueSystems, optimizer.target.defaultSuffix)

            if self.presentPareto is not None and len(self.presentPareto) == 2:
                self._drawParetoFronts2(fronts, optimizer)

    def _drawProperties(self, uniqueSystems):
        uniqueSystems = [uniqueSystems.getEntry(ID) for ID in uniqueSystems.getIDs()]
        for type, *arguments in self.toDraw:
            if type == 'dep':
                propertyY, typeY, propertyX, typeX = arguments
                suffixX = propertyX.split('.')[-1]
                suffixY = propertyY.split('.')[-1]
                Y = []
                X = []
                for system in uniqueSystems:
                    valueX = system[propertyX]
                    valueY = system[propertyY]
                    if typeY == 'raw':
                        Y.append(valueY)
                    elif typeY == 'per_atom':
                        Y.append(valueY/len(system[f'atomistic.molecules.{suffixY}']))
                    if typeX == 'raw':
                        X.append(valueX)
                    elif typeX == 'per_atom':
                        X.append(valueX/len(system[f'atomistic.molecules.{suffixX}']))
                plt.figure()
                plt.plot(X,Y,'go')
                plt.ylabel(f'{propertyY}({typeY})')
                plt.xlabel(f'{propertyX}({typeX})')
                plt.savefig(self.RES_FOLDER/f'{propertyY}({typeY})_vs_{propertyX}({typeX}).svg')
                plt.close()
            elif type == 'stat':
                propertyY, typeY = arguments
                suffixY = propertyY.split('.')[-1]
                Y = []
                for system in uniqueSystems:
                    value = system[propertyY]
                    if not np.isinf(value):
                        if typeY == 'raw':
                            Y.append(value)
                        elif typeY == 'per_atom':
                            Y.append(value/len(system[f'atomistic.molecules.{suffixY}']))
                plt.figure()
                plt.hist(Y, len(Y)//10+1, facecolor='g', alpha=0.75)
                plt.savefig(self.RES_FOLDER/f'{propertyY}({typeY})_statistics.svg')
                plt.close()

    def _drawExtendedConvexHull2(self, compositionSpace, convexHull, extendedConvexHull, suffix):
        if convexHull:
            leftNumBlocks = np.asarray(compositionSpace.numBlocks(convexHull[0]['simpleMoleculeUtility.composition.origin']), dtype = float)
            leftNumBlocksTotal = np.sum(leftNumBlocks)
            leftNumBlocks /= leftNumBlocksTotal
            leftEnthalpy = convexHull[0][f'.enthalpy.{suffix}']/leftNumBlocksTotal
            rightNumBlocks = leftNumBlocks
            rightEnthalpy = leftEnthalpy
            for system in convexHull:
                numBlocks = np.asarray(compositionSpace.numBlocks(system['simpleMoleculeUtility.composition.origin']), dtype = float)
                numBlocksTotal = np.sum(numBlocks)
                numBlocks /= numBlocksTotal
                if numBlocks[1] < leftNumBlocks[1]:
                    leftNumBlocks = numBlocks
                    leftEnthalpy = system[f'.enthalpy.{suffix}'] / numBlocksTotal
                elif numBlocks[1] > rightNumBlocks[1]:
                    rightNumBlocks = numBlocks
                    rightEnthalpy = system[f'.enthalpy.{suffix}'] / numBlocksTotal
            Xch = []
            Ych = []
            for system in convexHull:
                numBlocks = np.asarray(compositionSpace.numBlocks(system['simpleMoleculeUtility.composition.origin']), dtype = float)
                numBlocksTotal = np.sum(numBlocks)
                numBlocks /= numBlocksTotal
                C = np.array([leftNumBlocks, rightNumBlocks])
                E = np.array([leftEnthalpy, rightEnthalpy])
                Enthalpy = system[f'.enthalpy.{suffix}']/numBlocksTotal - np.dot(np.linalg.lstsq(C.T, numBlocks)[0], E)
                Xch.append(numBlocks[1])
                Ych.append(Enthalpy)
            inds = np.argsort(Xch)
            Xch = np.asarray(Xch)[inds]
            Ych = np.asarray(Ych)[inds]
            X = []
            Y = []
            for ID in extendedConvexHull.getIDs():
                system = extendedConvexHull.getEntry(ID)
                numBlocks = np.asarray(compositionSpace.numBlocks(system['simpleMoleculeUtility.composition.origin']), dtype = float)
                numBlocksTotal = np.sum(numBlocks)
                numBlocks /= numBlocksTotal
                C = np.array([leftNumBlocks, rightNumBlocks])
                E = np.array([leftEnthalpy, rightEnthalpy])
                Enthalpy = system[f'.enthalpy.{suffix}']/numBlocksTotal - np.dot(np.linalg.lstsq(C.T, numBlocks)[0], E)
                if Enthalpy < self.rangeECH:
                    X.append(numBlocks[1])
                    Y.append(Enthalpy)
            np.savetxt(self.RES_FOLDER/'ExtendedConvexHull.csv', np.stack((X,Y), axis=-1), fmt='%6.3f', delimiter=',')
            plt.figure()
            plt.plot(X,Y,'go')
            plt.plot(Xch,Ych,'b-o')
            plt.ylabel('Enthalpy of formation (eV/atom)')
            components = []
            for block in compositionSpace.blocks:
                components.append(''.join(f'{symbol}{mult if mult != 1 else ""}' \
                    for symbol, mult in zip(compositionSpace.symbols, block) if mult != 0))
            plt.xlabel(f'Composition ratio: {components[1]}/({components[0]}+{components[1]})')
            plt.savefig(self.RES_FOLDER/'ExtendedConvexHull.svg')
            plt.close()

    def _drawExtendedConvexHull3(self, compositionSpace, convexHull, extendedConvexHull, suffix):
        pass

    def _drawParetoFronts2(self, fronts, optimizer):
        pool = optimizer.generations[-1].goodSystems
        xProp = pool.createExpression(self.presentPareto[0])
        yProp = pool.createExpression(self.presentPareto[1])
        xLabel = getPresetLables(xProp)
        yLabel = getPresetLables(yProp)
        data = []
        for front in fronts:
            values = np.asarray([(system[xProp], system[yProp]) for system in front], dtype=float)
            data.append(values[np.argsort(values[:, 0])])
        plt.figure()
        for values, c in zip_longest(data, ['k-o', 'b-o', 'r-o', 'm-o', 'c-o'], fillvalue='go'):
            plt.plot(*values.T, c)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.savefig(self.RES_FOLDER/f'Pareto_{xLabel}_{yLabel}.svg')
        plt.close()

    @staticmethod
    def applyPresetOutputParameters(optimizer):
        columns = [column.split('.')[1] for column in  AtomisticRepresentation._extract(optimizer.optType)]
        if len(columns) > 1:
            presentPareto = columns
        else:
            presentPareto = None
        if 'enthalpyCCH' in columns or 'enthalpyCS' in columns:
            columns = ['enthalpy'] + columns
        if not 'simpleMoleculeUtility.composition' in columns:
            columns = ['simpleMoleculeUtility.composition'] + columns
        dim = optimizer.target.utilities.cellUtility.getDim()
        if dim == 3:
            if not 'cellUtility.volume' in columns:
                columns.append('cellUtility.volume')
            if not 'cellUtility.symmetry' in columns:
                columns.append('cellUtility.symmetry')
        elif dim == 2:
            if not 'cellUtility.area' in columns:
                columns.append('cellUtility.area')
        elif dim == 1:
            if not 'cellUtility.length' in columns:
                columns.append('cellUtility.length')
        elif dim == 0:
            pass
        else:
            raise RuntimeError(f'Wrong dim {dim}.')
        if not 'radialDistributionUtility.structureOrder' in columns:
            columns.append('radialDistributionUtility.structureOrder')
        if not 'radialDistributionUtility.averageOrder' in columns:
            columns.append('radialDistributionUtility.averageOrder')
        if not 'radialDistributionUtility.quasientropy' in columns:
            columns.append('radialDistributionUtility.quasientropy')
        if 'enthalpy' in columns:
            toDraw = [('dep', 'enthalpy', 'per_atom', 'ID', 'raw'),
                      ('stat', 'enthalpy', 'per_atom', '', '')]
            if 'enthalpyCCH' not in columns:
                toDraw.append(('dep', 'enthalpy', 'raw', 'ID', 'raw'))
            if 'cellUtility.volume' in columns:
                toDraw.append(('dep', 'enthalpy', 'per_atom', 'cellUtility.volume', 'per_atom'))
        else:
            toDraw = []
        presentConvexHull = 'enthalpyCCH' in columns
        for i, column in enumerate(columns):
            columns[i] = (column, presetLabels[column])
        if presentPareto is not None:
            for i, column in enumerate(presentPareto):
                presentPareto[i] = (column, presetLabels[column])
        return dict(
            columns=columns,
            toDraw=toDraw,
            presentConvexHull=presentConvexHull,
            presentPareto=presentPareto
        )

    @staticmethod
    def _extract(optType):
        if optType != 'enthalpyCCH' and optType != 'enthalpyCS' and optType in presetFitness:
            optType = presetFitness[optType]
        if isinstance(optType, str):
            return [optType]
        if isinstance(optType, tuple):
            func, *arguments = optType
            return list(set(chain(*[AtomisticRepresentation._extract(arg) for arg in arguments])))
        else:
            return []

