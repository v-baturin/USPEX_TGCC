import numpy as np
from os.path import join as pj
from ase.io.vasp import read_vasp_out, read_vasp_xml, write_vasp
from ase.io import ParseError, read
from ase.atoms import Atoms
from ase.constraints import FixAtoms


class ASEInterfaceAdapter:

    # VASP output files
    outcar_file = 'OUTCAR'
    oszicar_file = 'OSZICAR'
    contcar_file = 'CONTCAR'
    xml_file = 'vasprun.xml'

    # VASP input files
    incar_file = 'INCAR'
    kpoints_file = 'KPOINTS'
    poscar_file = 'POSCAR'
    potcar_file = 'POTCAR'

    # LAMMPS files
    data_file = 'STRUC'
    dump_file = 'lammps.dump'

    def __init__(self, atomType, cellType, structureType, **kwargs):
        self.atomType = atomType
        self.cellType = cellType
        self.structureType= structureType
        self.kwargs = kwargs

    def writeVASP(self, structure, fixedIndices, label, calcFolder):
        cell = structure.getCell()
        symbols = [el.short_name for el in structure.getAtomTypes()]
        atoms = Atoms(symbols, structure.getCartesianCoordinates(), cell=cell.getCellVectors())
        if fixedIndices:
            atoms.set_constraint(FixAtoms(indices=fixedIndices))
        with open(pj(calcFolder, self.poscar_file), 'wt') as f:
            write_vasp(f, atoms, label=label, sort=True, direct=True, vasp5=True, long_format=False)
        return {'pbc': cell.getPBC(), 'symbolsOrder': np.argsort(symbols)}

    def readVASP(self, calcFolder, targetProperties, pbc, symbolsOrder):
        try:
            atoms = read_vasp_out(pj(calcFolder, self.outcar_file))
        except (KeyError, ParseError):
            atoms = list(read_vasp_xml(pj(calcFolder, self.xml_file)))[-1]
        results = atoms.get_calculator().results

        data = {}
        if 'structure' in targetProperties:
            size = len(atoms)
            positions = np.empty((size, 3), dtype=float)
            atomTypes = np.empty(size, dtype=self.atomType)
            for i, symbol, position in zip(symbolsOrder, atoms.get_chemical_symbols(), atoms.get_positions()):
                positions[i] = position
                atomTypes[i] = self.atomType(symbol)
            cell = self.cellType(atoms.get_cell().array, pbc)
            structure = self.structureType(atomTypes, positions, cell=cell)
            data['structure'] = structure
        if 'energy' in targetProperties :
            data['energy'] = float(results['energy'])
        if 'enthalpy' in targetProperties:
            data['energy'] = float(results['energy'])
            data['volume'] = atoms.get_volume()
        if 'forces' in targetProperties:
            data['forces'] = np.copy(results['forces'])
        return data

    def writeLAMMPS(self, structure, fixedIndices, label, specorder, calcFolder):
        cell = structure.getCell()
        atoms = Atoms([el.short_name for el in structure.getAtomTypes()], structure.getCartesianCoordinates(),
                      cell=cell.getCellVectors())

        filename = pj(calcFolder, self.data_file)
        atoms.write(filename, format='lammps-data', specorder=specorder)
        with open(filename, 'rt') as f:
            content = f.readlines()
        content[0] = f'{label}\n'
        with open(filename, 'wt') as f:
            f.writelines(content)
        return {'pbc': cell.getPBC()}

    def readLAMMPS(self, calcFolder, targetProperties, specorder, pbc):
        atoms = read(pj(calcFolder, self.dump_file), format='lammps-dump-text')
        data = {}
        if 'structure' in targetProperties:
            atomTypes = np.array([self.atomType(specorder[i - 1]) for i in atoms.get_atomic_numbers()])
            data['structure'] = self.structureType(atomTypes, atoms.get_positions(),
                                                   cell=self.cellType(atoms.get_cell().array, pbc))
        return data
