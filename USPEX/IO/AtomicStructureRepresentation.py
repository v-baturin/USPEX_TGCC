import io
import shutil
import numpy as np
from ase.atoms import Atoms
from ase.io.vasp import write_vasp, read_vasp
from ase.io import read, write

from .read_molecule import read_molecule


class AtomicStructureRepresentation:

    structureType = None
    atomType = None
    cellType = None

    @classmethod
    def registerTypes(cls, structureType, atomType, cellType):
        cls.structureType = structureType
        cls.atomType = atomType
        cls.cellType = cellType

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
            types = [specorder[n] for n in types]
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
        atomTypes = [specorder.index(el.short_name) for el in structure.getAtomTypes()]
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
        with open(filename, "at") as f:
            shutil.copyfileobj(content, f)

    @classmethod
    def writePOSCAR(cls, filename, structure, label):
        structure = structure.getTrigonalizedCellStructure()
        coordinates = structure.getCartesianCoordinates()
        cell = structure.getCell().getEnvelopeCell(coordinates, 10)
        coordinates = cell.center(coordinates)
        atoms = Atoms([el.short_name for el in structure.getAtomTypes()], coordinates, cell=cell.getCellVectors())
        write_vasp(filename, atoms, label=label, direct=True, vasp5=True, long_format=False)

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
                    all_systems.append(cls.readPOSCAR(f))
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
    def readXYZ(cls, filename, extendedAtomTypes=None, **kwargs):
        atoms = read(filename, format='xyz')
        if extendedAtomTypes is not None:
            atomTypes = []
            columns = extendedAtomTypes['columns']
            data = extendedAtomTypes['data']
            symbols = atoms.get_chemical_symbols()
            assert len(data) == len(symbols)
            for symbol, values in zip(symbols, data):
                atomTypes.append(cls.atomType(symbol, **dict(zip(columns, values))))
        else:
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

    @staticmethod
    def toAtoms(structure) -> Atoms:
        cell = structure.getCell()
        return Atoms(symbols=[el.short_name for el in structure.getAtomTypes()],
                     positions=structure.getCartesianCoordinates(),
                     cell=cell.getCellVectors(), pbc=cell.getPBC())

    @classmethod
    def fromAtoms(cls, atoms: Atoms):
        cellVectors = atoms.get_cell().array
        pbc = atoms.get_pbc()
        if np.allclose(cellVectors, 0) and sum(pbc) == 0:
            cellVectors = np.eye(3)
        return cls.structureType(atomTypes=[cls.atomType(s) for s in atoms.get_chemical_symbols()],
                                 coordinates=atoms.get_positions(),
                                 cell=cls.cellType(cellVectors, pbc))
