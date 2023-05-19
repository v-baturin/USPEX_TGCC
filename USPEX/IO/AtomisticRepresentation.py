import io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import shutil
import yaml

from ase.atoms import Atoms
from ase.io.vasp import write_vasp, read_vasp
from ase.io import read, write
from copy import copy
from collections import Counter
from collections.abc import Mapping
from itertools import combinations, chain
from pathlib import Path
from prettytable import PrettyTable

from .formatters import createHeader_wrap
from ..presets import presetFitness
from .read_molecule import read_molecule

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
    'enthalpy': 'Enthalpy (eV)',
    'enthalpyCCH': 'Enthalpy above CH (eV/block)',
    'enthalpyCS': 'Enthalpy above the best for composition(eV/block)',
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
        if self.isRank:
            row.insert(1, rank)
        for column, columnName in self.columns:
            value = fitness.getFitnessByID(column, ID)
            if value is None:
                try:
                    value = fitness.getFitnessDirect(column, system)
                except Exception:
                    pass
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

    @classmethod
    def registerTypes(cls, structureType, atomType, cellType, atomicDisassemblerType):
        cls.structureType = structureType
        cls.atomType = atomType
        cls.cellType = cellType
        cls.atomicDisassemblerType = atomicDisassemblerType

    def __init__(self, RES_FOLDER: str, columns, toDraw, presentConvexHull: bool, presentPareto,
                 rangeECH = EXTENDED_CONVEX_HULL_ENERGY_RANGE, **kwargs):
        self.RES_FOLDER = Path(RES_FOLDER)
        self.columns = columns
        self.toDraw = toDraw
        self.presentConvexHull = presentConvexHull
        self.presentPareto = presentPareto
        self.rangeECH = rangeECH

    def getNewSystemsTable(self, isRank=False):
        return SystemsTable(self.columns, isRank)

    def presentSystems(self, systems: dict, optimizer, numStages):
        systems_gatheredPOSCARS = []
        systems_gatheredPOSCARS_unrelaxed = []
        table_Individuals = self.getNewSystemsTable()
        content_origin = ''
        content_enthalpies = ''
        for ID, system in sorted(systems.items()):
            systems_gatheredPOSCARS_unrelaxed.append(system[0])
            content_origin += f"{ID} {system[0]['howCome']} {system[0]['parent']}\n"

            if len(system) > 1:
                content_enthalpies += ','.join([f"{sys['enthalpy']:6.3f}" for sys in system[1:]]) + '\n'

            if len(system) == numStages + 1:
                table_Individuals.update(ID, system[-1], optimizer.fitness)
                systems_gatheredPOSCARS.append(system[numStages])

        self.RES_FOLDER.mkdir(parents=True, exist_ok=True)

        self.writeAtomicStructures(self.RES_FOLDER/'gatheredPOSCARS_unrelaxed',
                                   systems_gatheredPOSCARS_unrelaxed)
        self.writeAtomicStructures(self.RES_FOLDER/'gatheredPOSCARS',
                                   systems_gatheredPOSCARS)
        with open(self.RES_FOLDER/'Individuals', 'w') as f:
            f.write(table_Individuals.table.get_string() + '\n')
        with open(self.RES_FOLDER/'origin', 'w') as f:
            f.write(content_origin)
        with open(self.RES_FOLDER/'enthalpies_complete.csv', 'w') as f:
            f.write(content_enthalpies)
        self.drawESeries(systems, numStages)

    def drawESeries(self, systems, numStages):
        enths = []
        for system_stages in systems.values():
            if len(system_stages) == numStages + 1:
                enths.append([system['enthalpy'] for system in system_stages[1:]])
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
    def readMLIPcfg(cls, file, specorder=None):
        lat = np.zeros((3, 3))
        types = None
        pos = None
        energy = None
        forces = None
        stresses = None
        size = -1
        mode = -1
        line = file.readline()
        while line:
            line = line.upper()
            line = line.strip()
            if mode == 0:
                if line.startswith('SIZE'):
                    line = file.readline()
                    size = int(line.strip())
                    types = np.zeros(size, dtype=int).tolist()
                    pos = np.zeros((size, 3))
                elif line.startswith('SUPERCELL'):
                    line = file.readline()
                    vals = line.strip().split()
                    lat[0, :] = vals[0:3]
                    line = file.readline()
                    vals = line.strip().split()
                    lat[1, :] = vals[0:3]
                    line = file.readline()
                    vals = line.strip().split()
                    lat[2, :] = vals[0:3]
                elif line.startswith('ATOMDATA'):
                    if line.endswith('FZ'):
                        forces = np.zeros((size, 3))
                    for i in range(size):
                        line = file.readline()
                        vals = line.strip().split()
                        types[i] = int(vals[1])
                        pos[i, :] = vals[2:5]
                        if forces is not None:
                            forces[i, :] = vals[5:8]
                elif line.startswith('ENERGY'):
                    line = file.readline()
                    energy = float(line.strip())
                elif line.startswith('PLUSSTRESS'):
                    line = file.readline()
                    vals = line.strip().split()
                    stresses = np.zeros(6)
                    stresses[:] = vals[0:6]
            if line.startswith('BEGIN_CFG'):
                mode = 0
            elif line.startswith('END_CFG'):
                break
            line = file.readline()

        cell = cls.cellType(lat, (1, 1, 1))
        if specorder is not None:
            types = [specorder[n-1] for n in types]
        return dict(
            structure=cls.structureType([cls.atomType(n) for n in types], pos, cell=cell),
            energy=energy,
            forces=forces,
            stresses=stresses
        )

    @classmethod
    def readMLIPsample(cls, filename, specorder):
        all_systems = []
        with open(filename, 'r') as f:
            while True:
                try:
                    all_systems.append(cls.readMLIPcfg(f, specorder))
                except Exception:
                    break
        return all_systems

    @staticmethod
    def saveMLIPcfg(f, specorder, structure, forces=None, energy=None, stresses=None, **kwargs):
        atstr1 = 'AtomData:  id type      cartes_x      cartes_y      cartes_z           fx          fy          fz\n'
        atstr2 = 'AtomData:  id type      cartes_x      cartes_y      cartes_z\n'
        size = len(structure)
        f.write('BEGIN_CFG\n')
        f.write('Size\n')
        f.write(f'   {size}\n')
        f.write('SuperCell\n')
        for i in range(3):
            lat = structure.getCell().getCellVectors()
            f.write(' %13f %13f %13f\n' % (lat[i, 0], lat[i, 1], lat[i, 2]))
        if forces is not None:
            f.write(atstr1)
        else:
            f.write(atstr2)
        atomTypes =  [specorder.index(el.short_name) for el in structure.getAtomTypes()]
        positions = structure.getCartesianCoordinates()
        for i in range(size):
            if forces is not None:
                f.write('         %4d %4d %13f %13f %13f %11.8e %11.8e %11.8e\n' %
                        (i + 1, atomTypes[i], positions[i, 0], positions[i, 1], positions[i, 2],
                         forces[i, 0], forces[i, 1], forces[i, 2]))
            else:
                f.write('         %4d %4d %13f %13f %13f\n' %
                        (i + 1, atomTypes[i], positions[i, 0], positions[i, 1], positions[i, 2]))
        if energy is not None:
            f.write(' Energy\n   %20f\n' % energy)
        if stresses is not None:
            f.write(' PlusStress:  xx           yy           zz           yz           xz           xy\n')
            f.write('         %11f %11f %11f %11f %11f %11f\n' %
                    (stresses[0], stresses[1], stresses[2],
                     stresses[3], stresses[4], stresses[5]))
        f.write('END_CFG\n')

    @classmethod
    def saveMLIPsample(cls, filename, specorder, sample):
        content = io.StringIO('')
        for system in sample:
            cls.saveMLIPcfg(content, specorder, **system)
        content.seek(0)
        with open(filename, "wt") as f:
            shutil.copyfileobj(content, f)

    @classmethod
    def writePOSCAR(cls, filename, structure, label):
        structure = structure.getTrigonalizedCellStructure()
        coordinates = structure.getCartesianCoordinates()
        cell = structure.getCell().getEnvelopeCell(coordinates, 10)
        coordinates = cell.center(coordinates)
        atoms = Atoms([el.short_name for el in structure.getAtomTypes()], coordinates, cell=cell.getCellVectors())
        write_vasp(filename, atoms, label=label, sort=True, direct=True, vasp5=True, long_format=False)

    @classmethod
    def writePOSCARS(cls, filename, structures, labels):
        content = io.StringIO('')
        for structure, label in zip(structures, labels):
            cls.writePOSCAR(content, structure, label)
        content.seek(0)
        with open(filename, "wt") as f:
            shutil.copyfileobj(content, f)

    @classmethod
    def writeXYZ(cls, filename, structure, label=''):
        coordinates = structure.getCartesianCoordinates()
        atoms = Atoms([el.short_name for el in structure.getAtomTypes()], coordinates, cell=None)
        write(filename, atoms, format='xyz', comment=label)

    @classmethod
    def writeAtomicStructure(cls, filename, system: dict):
        filename = Path(filename)
        cls.writeAtomicStructures(filename, [system])

    @classmethod
    def writeAtomicStructures(cls, filename: Path, systems: list):
        structures = []
        labels = []
        descriptions = []
        printUSPEX = False
        for i, system in enumerate(systems):
            structure, disassembler = cls.atomicDisassemblerType.assemble(**system, vacuumSize=10.0)
            atomTypes = structure.getAtomTypes()
            coordinates = structure.getCartesianCoordinates()
            sortIndices = np.argsort(atomTypes)
            reversedIndices = np.argsort(sortIndices)
            structure = cls.structureType(atomTypes[sortIndices], coordinates[sortIndices], structure.getCell())
            structures.append(structure)
            labels.append(f"EA{system['ID']}")
            d = {'filename': filename.name, 'index': i}
            pbc = system['cell'].getPBC()
            if pbc != (1, 1, 1):
                d['pbc'] = ' '.join(f'{c}' for c in pbc)
                printUSPEX = True
            molecules = []
            for indices in disassembler.indices:
                if len(indices) > 1:
                    molecules.append(' '.join(f'{ind}' for ind in reversedIndices[indices]))
                    printUSPEX = True
            if molecules:
                d['molecules'] = molecules
            if 'environments' in system:
                printUSPEX = True
                d['environments'] = []
                for eInds in disassembler.envIndices:
                    d['environments'].append(' '.join(f'{ind}' for ind in eInds))
                d['fixed'] = ' '.join(f'{ind}' for ind in disassembler.allFixedIndices)
            descriptions.append(d)
        cls.writePOSCARS(filename, structures, labels)
        if printUSPEX:
            with open(f'{filename}.uspex', 'wt') as f:
                f.write(yaml.safe_dump(descriptions))


    @classmethod
    def readPOSCAR(cls, filename, pbc=(1, 1, 1)):
        atoms = read_vasp(filename)
        atomTypes = [cls.atomType(s) for s in atoms.get_chemical_symbols()]
        cell = cls.cellType(atoms.get_cell().array, pbc)
        coordinates = atoms.get_positions()
        return cls.structureType(atomTypes, coordinates, cell)

    @classmethod
    def readPOSCARS(cls, filename):
        all_systems = []
        with open(filename, 'rt') as f:
            while True:
                try:
                    all_systems.append(AtomisticRepresentation.readPOSCAR(f))
                except Exception:
                    break
        return all_systems

    @classmethod
    def readMol(cls, filename):
        molDct = read_molecule(filename)
        atomTypes = [cls.atomType(s) for s in molDct['symbols']]
        coordinates = molDct['positions']
        zmatrixConfig = molDct['configZMatrix']
        return cls.structureType(atomTypes, coordinates, zmatrixConfig=zmatrixConfig)


    @classmethod
    def readXYZ(cls, filename):
        atoms = read(filename, format='xyz')
        atomTypes = [cls.atomType(s) for s in atoms.get_chemical_symbols()]
        coordinates = atoms.get_positions()
        cell = cls.cellType.initFromCellParameters((0, 0, 0)).getEnvelopeCell(coordinates)
        return cls.structureType(atomTypes, coordinates, cell)

    @classmethod
    def readXYZs(cls, filename):
        all_atoms = read(filename, index=':', format='xyz')
        all_systems = []
        dummy_cell = cls.cellType.initFromCellParameters((0, 0, 0))
        for atoms in all_atoms:
            all_systems.append(cls.structureType([cls.atomType(s) for s in atoms.get_chemical_symbols()],
                                                 atoms.get_positions(),
                                                 dummy_cell.getEnvelopeCell(atoms.get_positions())))
        return all_systems

    @classmethod
    def readAtomicStructure(cls, filename) -> dict:
        return cls.readAtomicStructures(filename)[0]

    @classmethod
    def readAtomicStructures(cls, filename) -> list:
        filename = Path(filename)
        directory = filename.parent
        if filename.suffix == '.uspex':
            with open(filename) as f:
                descriptions = yaml.safe_load(f.read())
            files = {name: cls.readPOSCARS(directory/name)
                     for name in np.unique([s['filename'] for s in descriptions])}
            systems = []
            for d in descriptions:
                d = copy(d)
                structure = files[d.pop('filename')][d.pop('index')]
                if 'pbc' in d:
                    d['pbc'] = tuple(int(c) for c in d.pop('pbc').split(' '))
                if 'molecules' in d:
                    d['indices'] = [np.array(mol.split(' '), dtype=int) for mol in d.pop('molecules')]
                else:
                    d['indices'] = []
                fixed = np.array(d.pop('fixed').split(' '), dtype=int) if 'fixed' in d else np.empty(0, dtype=int)
                if 'environments' in d:
                    d['envIndices'] = []
                    d['fixedIndices'] = []
                    for eInds in d.pop('environments'):
                        eInds = np.array(eInds.split(' '), dtype=int)
                        fInds = np.argwhere(eInds.reshape((-1, 1)) == fixed.reshape((1, -1)))[:, 0]
                        d['envIndices'].append(eInds)
                        d['fixedIndices'].append(fInds)
                    envIndices = np.concatenate(d['envIndices'])
                else:
                    envIndices = []
                d['indices'].extend([np.array([i]) for i in set(range(len(structure))).difference(set(envIndices))])
                systems.append(cls.atomicDisassemblerType(**d).disassemble(structure))
        else:
            systems = [cls.atomicDisassemblerType(np.arange(len(structure)).reshape((-1, 1))).disassemble(structure)
                       for structure in cls.readPOSCARS(filename)]
        return systems

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
            if not s1['isBad'] and not s2['isBad']:
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
        systems__BESTgatheredPOSCARS = []
        systems_goodStructuresPOSCARS = []
        systems_extendedConvexHullPOSCARS = []

        self.RES_FOLDER.mkdir(parents=True, exist_ok=True)

        fitness = optimizer.optType

        for generation, opt in enumerate(optimizers):
            content_BESTIndividuals += f'Generation {generation}\n'
            pool = opt.pool
            table = self.getNewSystemsTable()
            for ID in opt.best:
                table.update(ID, pool.allSystems[ID], opt.fitness)
            content_BESTIndividuals += table.table.get_string() + '\n'
        with open(self.RES_FOLDER/'BESTIndividuals', 'w') as fp:
            fp.write(content_BESTIndividuals)

        for opt in optimizers:
            pool = opt.pool
            for ID in opt.best:
                systems__BESTgatheredPOSCARS.append(pool.allSystems[ID])
        self.writeAtomicStructures(self.RES_FOLDER/'BESTgatheredPOSCARS', systems__BESTgatheredPOSCARS)

        compositionSpace = optimizer.target.utilities.compositionSpace
        csSize = len(compositionSpace.blocks)

        allFitnesses = optimizer.fitness.getAllFitnesses(fitness)
        fronts = optimizer.fitness.sort(list(optimizer.pool.uniqueSystems), allFitnesses)
        if csSize == 1:
            for rank, front in enumerate(fronts):
                for system in front:
                    table_goodStructures.update(system['ID'], system, optimizer.fitness, rank=rank)
                    systems_goodStructuresPOSCARS.append(system)
            with open(self.RES_FOLDER/'goodStructures', 'w') as fp:
                fp.write(table_goodStructures.table.get_string() + '\n')

            self.writeAtomicStructures(self.RES_FOLDER/'goodStructures_POSCARS', systems_goodStructuresPOSCARS)
        else:
            goodStructresFolder = self.RES_FOLDER/'goodStructures'
            goodStructresFolder.mkdir(parents=True, exist_ok=True)
            goodStructures = {}
            goodStructuresPOSCARS = {}
            for rank, front in enumerate(fronts):
                for system in front:
                    numBlocks = tuple(compositionSpace.numBlocks(system['simpleMoleculeUtility.composition']))
                    if numBlocks not in goodStructures:
                        goodStructures[numBlocks] = self.getNewSystemsTable(isRank=True)
                        goodStructuresPOSCARS[numBlocks] = []
                    goodStructures[numBlocks].update(system['ID'], system, optimizer.fitness, rank=rank)
                    goodStructuresPOSCARS[numBlocks].append(system)

            for comp, table_gs in goodStructures.items():
                with open(goodStructresFolder/f'{"_".join(str(x) for x in comp)}', 'w') as fp:
                    fp.write(table_gs.table.get_string() + '\n')

            for comp, systems_gs_POSCARS in goodStructuresPOSCARS.items():
                self.writeAtomicStructures(goodStructresFolder/f'{"_".join(str(x) for x in comp)}_POSCARS',
                                           systems_gs_POSCARS)

        if self.presentConvexHull:
            convexHull = []
            for generation, opt in enumerate(optimizers):
                convexHull = [system for system in opt.pool.uniqueSystems
                              if np.isclose(opt.fitness.getFitnessByID('enthalpyCCH', originalID(system)), 0.0)]
                content_convexHull += f'Generation {generation}\n'
                table = self.getNewSystemsTable()
                for system in convexHull:
                    table.update(system['ID'], system, opt.fitness)
                content_convexHull += table.table.get_string() + '\n'

            with open(self.RES_FOLDER/'convex_hull', 'w') as fp:
                fp.write(content_convexHull)

            extendedConvexHull = [system for system in optimizer.pool.uniqueSystems
                                  if optimizer.fitness.getFitnessByID('enthalpyCCH', originalID(system)) < self.rangeECH]

            allFitnesses = {system['ID']: optimizer.pool.generations[-1]['fitness'].getFitnessByID(fitness, originalID(system))
                            for system in extendedConvexHull}
            frontsECH = optimizer.fitness.sort(extendedConvexHull, allFitnesses)

            for rank, front in enumerate(frontsECH):
                for system in front:
                    table_extendedConvexHull.update(system['ID'], system, optimizer.fitness, rank=rank)
            with open(self.RES_FOLDER/'extended_convex_hull', 'w') as fp:
                fp.write(table_extendedConvexHull.table.get_string())

            for front in frontsECH:
                for system in front:
                    systems_extendedConvexHullPOSCARS.append(system)
            self.writeAtomicStructures(self.RES_FOLDER/'extended_convex_hull_POSCARS',
                                       systems_extendedConvexHullPOSCARS)

            if csSize == 2:
                self._drawExtendedConvexHull2(compositionSpace, convexHull + optimizer.extraData, extendedConvexHull)
            elif csSize == 3:
                self._drawExtendedConvexHull3(compositionSpace, convexHull + optimizer.extraData, extendedConvexHull)

        if self.presentPareto is not None and len(self.presentPareto) == 2:
            self._drawParetoFronts2(fronts, optimizer)

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
                plt.figure()
                plt.plot(X,Y,'go')
                plt.ylabel(f'{propertyY}({typeY})')
                plt.xlabel(f'{propertyX}({typeX})')
                plt.savefig(self.RES_FOLDER/f'{propertyY}({typeY})_vs_{propertyX}({typeX}).svg')
                plt.close()
            elif type == 'stat':
                Y = []
                for system in uniqueSystems:
                    value = fitness.getFitnessByID(propertyY, originalID(system))
                    if not np.isinf(value):
                        if typeY == 'raw':
                            Y.append(value)
                        elif typeY == 'per_atom':
                            Y.append(value/len(system['molecules']))
                plt.figure()
                plt.hist(Y, len(Y)//10+1, facecolor='g', alpha=0.75)
                plt.savefig(self.RES_FOLDER/f'{propertyY}({typeY})_statistics.svg')
                plt.close()

    def _drawExtendedConvexHull2(self, compositionSpace, convexHull, extendedConvexHull):
        if convexHull:
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


    def _drawExtendedConvexHull3(self, compositionSpace, convexHull, extendedConvexHull):
        pass
    def _drawParetoFronts2(self, fronts, optimizer):
        (xProp, xLabel), (yProp, yLabel) = self.presentPareto
        plt.figure()
        for front, c in zip(fronts, ['k', 'b', 'r', 'm', 'c']):
            values = np.asarray([(optimizer.fitness.getFitnessByID(xProp, system['ID']),
                                  optimizer.fitness.getFitnessByID(yProp, system['ID'])) for system in front],
                                dtype=float)
            values = values[np.argsort(values[:, 0])]
            plt.plot(*values.T, f'{c}-o')
        if len(fronts) > 5:
            for front in fronts[5:]:
                values = np.asarray([(optimizer.fitness.getFitnessByID(xProp, system['ID']),
                                      optimizer.fitness.getFitnessByID(yProp, system['ID'])) for system in front],
                                    dtype=float)
                values = values[np.argsort(values[:, 0])]
                plt.plot(*values.T, 'go')
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.savefig(self.RES_FOLDER/f'Pareto_{xProp}_{yProp}.svg')
        plt.close()

    @staticmethod
    def applyPresetOutputParameters(optimizer):
        columns = AtomisticRepresentation._extract(optimizer.optType)
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

