import numpy as np
from ase.io.vasp import iread_vasp_out, read_vasp_xml, write_vasp
from ase.io.espresso import read_fortran_namelist, read_espresso_out, write_espresso_in
from ase.io import ParseError, read
from ase.atoms import Atoms
from ase.constraints import FixAtoms
from ase.io.gen import read_gen, write_gen
from pathlib import Path


class ASEInterfaceAdapter:

    structureType = None
    atomType = None
    cellType = None

    @classmethod
    def registerTypes(cls, structureType, atomType, cellType):
        cls.structureType = structureType
        cls.atomType = atomType
        cls.cellType = cellType

    class Results:

        EV_PER_CUBIC_ANGSTREM_PER_GPA = 1 / 160.21766208

        def __init__(self, atoms):
            self.results = atoms.get_calculator().results
            self.volume = atoms.get_volume()

        def getEnthalpy(self, pressure):
            return self.results['enthalpy'] if 'enthalpy' in self.results else\
                self.results['energy'] + self.volume * pressure * self.EV_PER_CUBIC_ANGSTREM_PER_GPA

    class VASP:

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

        def write(self, structure, fixedIndices, label, calcFolder: Path):
            cell = structure.getCell()
            symbols = np.asarray([el.short_name for el in structure.getAtomTypes()])
            order = np.argsort(symbols)
            atoms = Atoms(symbols[order], structure.getCartesianCoordinates()[order], cell=cell.getCellVectors())
            if len(fixedIndices) > 0:
                atoms.set_constraint(FixAtoms(indices=fixedIndices))
            write_vasp(calcFolder/self.poscar_file, atoms, label=label, direct=True, vasp5=True, long_format=False)
            return {'pbc': cell.getPBC(), 'symbolsOrder': order}

        def read(self, calcFolder: Path, pbc, symbolsOrder):
            try:
                with open(calcFolder/self.outcar_file) as f:
                    trajectoryAtoms = list(iread_vasp_out(f, None))
            except (KeyError, ParseError):
                with open(calcFolder/self.xml_file) as f:
                    trajectoryAtoms = list(read_vasp_xml(f))
            trajectory = []
            for atoms in trajectoryAtoms:
                size = len(atoms)
                positions = np.empty((size, 3), dtype=float)
                atomTypes = np.empty(size, dtype=ASEInterfaceAdapter.atomType)
                for i, symbol, position in zip(symbolsOrder, atoms.get_chemical_symbols(), atoms.get_positions()):
                    positions[i] = position
                    atomTypes[i] = ASEInterfaceAdapter.atomType(symbol)
                cell = ASEInterfaceAdapter.cellType(atoms.get_cell().array, pbc)
                structure = ASEInterfaceAdapter.structureType(atomTypes, positions, cell=cell)
                trajectory.append(dict(
                    structure=structure,
                    results=ASEInterfaceAdapter.Results(atoms)
                ))
            return trajectory

    class LAMMPS:

        # LAMMPS files
        data_file = 'STRUC'
        dump_file = 'lammps.dump'

        def write(self, structure, fixedIndices, label, specorder, calcFolder):
            cell = structure.getCell()
            atoms = Atoms([el.short_name for el in structure.getAtomTypes()], structure.getCartesianCoordinates(),
                          cell=cell.getCellVectors())
            filename = calcFolder/self.data_file
            atoms.write(filename, format='lammps-data', specorder=specorder)
            with open(filename, 'rt') as f:
                content = f.readlines()
            content[0] = f'{label}\n'
            with open(filename, 'wt') as f:
                f.writelines(content)
            return {'pbc': cell.getPBC()}

        def read(self, calcFolder: Path, specorder, pbc):
            if calcFolder.joinpath(self.dump_file).exists():
                atoms = read(calcFolder/self.dump_file, format='lammps-dump-text')
                atomTypes = np.array([ASEInterfaceAdapter.atomType(specorder[i - 1]) for i in atoms.get_atomic_numbers()])
                structure = ASEInterfaceAdapter.structureType(atomTypes, atoms.get_positions(),
                                                              cell=ASEInterfaceAdapter.cellType(atoms.get_cell().array, pbc))
                return dict(
                    structure=structure,
                    results=ASEInterfaceAdapter.Results(atoms)
                )
            else:
                return None

    class QE:

        inputFile, outputFile = 'input', 'output'

        def __init__(self, options):
            with open(options) as fp:
                data, card_lines = read_fortran_namelist(fp)
            if 'system' not in data:
                raise KeyError('Required section &SYSTEM not found.')
            self.data = data

        def write(self, structure, fixedIndices, kPoints, pseudopotentials, calcFolder: Path):
            cell = structure.getCell()
            atoms = Atoms(symbols=[el.short_name for el in structure.getAtomTypes()],
                          positions=structure.getCartesianCoordinates(),
                          cell=cell.getCellVectors())
            if len(fixedIndices) > 0:
                atoms.set_constraint(FixAtoms(indices=fixedIndices))
            with open(calcFolder/self.inputFile, 'wt') as f:
                write_espresso_in(f,
                                  atoms=atoms, input_data=self.data,
                                  pseudopotentials={s: p.name for s, p in pseudopotentials.items()},
                                  kpts=kPoints,
                                  crystal_coordinates=True)
            return {'pbc': cell.getPBC()}

        def read(self, calcFolder: Path, pbc):
            with open(calcFolder/self.outputFile) as f:
                atoms = next(read_espresso_out(f, index=slice(None, -2, -1)))
            atomTypes = np.array([ASEInterfaceAdapter.atomType(s) for s in atoms.get_chemical_symbols()])
            structure = ASEInterfaceAdapter.structureType(atomTypes, atoms.get_positions(),
                                                          cell=ASEInterfaceAdapter.cellType(atoms.get_cell().array, pbc))
            return dict(
                structure=structure,
                results=ASEInterfaceAdapter.Results(atoms)
            )

    class GEN:

        def write_structure(self, structure, geometry_file, calcFolder: Path):
            cell = structure.getCell()
            pbc = cell.getPBC()
            symbols = np.asarray([el.short_name for el in structure.getAtomTypes()])
            coordinates = structure.getCartesianCoordinates()
            cell_vectors = cell.getCellVectors()
            ase_struct = Atoms(symbols, positions=coordinates, cell=cell_vectors, pbc=pbc)
            write_gen(calcFolder / geometry_file, ase_struct)
            return {'pbc': cell.getPBC()}

        def read_structure(self, geometry_file, calcFolder: Path, pbc):
            ase_struct = read_gen(calcFolder / geometry_file)
            new_lattice = []
            tmp_lattice = ase_struct.cell[:].copy()
            for i, vec in enumerate(tmp_lattice):
                if pbc[i]:
                    new_lattice.append([float(x) for x in vec])
            cell = ASEInterfaceAdapter.cellType.initFromCellVectors(pbc, new_lattice)
            new_structure = ASEInterfaceAdapter.structureType([ASEInterfaceAdapter.atomType(i)
                                                               for i in ase_struct.get_chemical_symbols()],
                                                              ase_struct.get_positions(), cell=cell)

            return new_structure