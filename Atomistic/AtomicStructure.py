"""
USPEX.Common.Atomistic.AtomicStructure
======================================

Class Atoms-type structure with properties and without periodicity (not Crystal)

.. codeauthor:: Artem Samtsevich <samtsevichartem@gmail.com>
.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

from ase.atoms import Atoms

import copy
import numpy as np
from typing import Dict, List, Tuple
from itertools import combinations_with_replacement, chain
from collections import Counter


from ..System import System
from .optLattice import optLattice
from .Element import Element
from .calcDefaultVolume import calcVolume
from .CompositionSpace import Composition
from .mol.coord2Zmatrix import coord2Zmatrix
from .mol.zmatrix2coord import zmatrix2coord
from .mol.find_pair import find_pair
from .Fingerprints.make_matrices import make_matrices
from .Fingerprints.fingerprint import fingerprint
from .Fingerprints.cosine_distance import cosine_distance
from .Fingerprints.quasientropy import quasientropy
from .Fingerprints.structure_order import structure_order


RMAX_DEFAULT = 10.0
SIGMA_DEFAULT = 0.03
DELTA_DEFAULT = 0.08

TOLERANCE_DEFAULT = 0.008


class AtomicStructure(System):
    """
    Class describing generic atoms-composed structure with properties.

    It is based on (but not inherited from) ase.Atoms.
    It exposes all non special methods of ase.Atoms. These methods are not listed here,
    see documentation of ase.Atoms for details. Special methods, like list-methods, are overridden.

    Besides ase.Atoms functional class, it provides tools for creating and manipulating molecular structures.
    """

    def __init__(self, molecules: list=[], optimizeLattice: bool=False, volumeType: str=None,
                 ionDistances: Dict[Tuple[str, str], float] = None,
                 goodBonds: Dict[Tuple[str, str], float] = None, valences: Dict[str, float] = None,
                 valenceElectrons: Dict[str, float] = None,
                 fingerprints: Dict[str, float] = None,
                 externalPressure: float = 0.0001,
                 **kwargs):
        """
        :type molecules: list of :class:`AtomicStructure`
        :param molecules: Supposed to be list of molecules. Deprecated. Will be removed soon.
        :type optimizeLattice: bool
        :param optimizeLattice: If True we will call :meth:`optimizeLattice` right after construction. If False we wont.
        :type externalPressure: float
        :param externalPressure: External pressure for this structure in GPa.
        :type kwargs: dict
        :param kwargs: parameters forwarded to ase.Atoms.

        .. Warning:: Initialization by list of molecules is deprecated in favor of alternative constructor
          :meth:`AtomicStructure.fromStructureList`.
        """
        self._molecules = []
        self.format = []
        self.flex_dihedral = []
        self.molSymbol = []

        self.atoms = Atoms(**kwargs)
        if optimizeLattice:
            self.optimizeLattice()

        self.config = {}
        if volumeType is not None:
            self.config['volumeType'] = volumeType
        if ionDistances is not None:
            self.config['ionDistances'] = ionDistances
        if goodBonds is not None:
            self.config['goodBonds'] = goodBonds
        if valences is not None:
            self.config['valences'] = valences
        if valenceElectrons is not None:
            self.config['valenceElectrons'] = valenceElectrons
        if fingerprints is not None:
            self.config['fingerprints'] = fingerprints

        symbols = self.atoms.get_chemical_symbols()
        self._molecules = [[i] for i in range(len(symbols))]
        self.format = [[[0, 0, 0]] for s in symbols]
        self.flex_dihedral = [[] for s in symbols]
        self.molSymbol = copy.copy(symbols)

        assert (not symbols) or (not molecules)     # Cannot initialize with both

        self._moleculeTypesToFormula = {}

        for molecule in molecules:  # This is deprecated and will be removed
            self.extend(molecule)

        self.externalPressure = externalPressure
        self._fingerprint = {}
        self._atomFingerprint = []
        self._order = []
        self._fingerprintWeights = {}
        self.fingerprintTolerance = fingerprints['tolerance'] if fingerprints is not None and'tolerance' in fingerprints\
            else TOLERANCE_DEFAULT

        self._goodBonds = {}
        self._valences = {}
        self._valenceElectrons = {}
        self._mDM = None

        self.energy = np.inf
        self.enthalpy = np.inf
        self._forces = None
        self._pressureTensor = None
        self._dielectricTensor = None
        self.bonds = None

        super().__init__()

    @classmethod
    def fromStructureList(cls, structures : list, **kwargs):
        """
        Alternative constructor for :class:`AtomicStructure` and its descendants.
        First creates an empty structure with the usual constructor and with *kwargs* as parameters
        and then extends it via :meth:`extend` with structures from *structures* list.

        :type structures: list of :class:`AtomicStructure`
        :param structures: list of structures to compose the new structure.
        :type kwargs: dict
        :param kwargs: parameters for initial empty structure construction.
        :rtype: type corresponding to this classmethod. (see Examples)
        :return: new created structure.

        :examples:

        Create aperiodic structure of type :class:`AtomicStructure` with three molecules: m1, m2, m3.

        >>> newstructure = AtomicStructure.fromStructureList([m1,m2,m3], pbs = False)

        Create structure of type **Crystal** with two molecules: m1, m2
        and cell vectors [1.0,0.0,0.0], [0.0,1.0,0.0] and [0.0,0.0,1.0].
        Assuming **Crystal** is subclass of :class:`AtomicStructure`.

        >>> newstructure = Crystal.fromStructureList([m1,m2], cell = [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])

        """
        assert 'symbols' not in kwargs
        newstructure = cls(**kwargs)
        for structure in structures:
            newstructure.extend(structure)
        return newstructure

    def __getattr__(self, item):
        """
        Special method for forwarding attributes which are not explicitly defined in AtomicStructure to ase.atoms class.

        :param item: Attribute name
        """
        if item == '__setstate__' or item == 'atoms' or not hasattr(self, 'atoms'):
            raise AttributeError(self, item)
        elif hasattr(self.atoms, item):
            return getattr(self.atoms, item)
        else:
            raise AttributeError(self,item)

    def __len__(self):
        """
        Special method which returns the number of atoms in the structure.
        """
        return len(self.atoms)

    def extend(self, ext):
        """
        Extend the current system with a given one.
        This is done by adding molecules from given structure to current one.
        Geometries of all molecules are kept. Cell parameters of the current structure are kept.
        New molecules will have the same scaled coordinates of their centers as in given structure.

        :type ext: :class:`AtomicStructure` or descendant
        :param ext:  a structure to extend the current one.
        """
        for inds, frmt, flex_dihedral, molSymbol in zip(ext._molecules, ext.format, ext.flex_dihedral, ext.molSymbol):
            inds = (np.asarray(inds) + len(self.atoms)).tolist()
            self._molecules.append(inds)
            self.format.append(frmt)
            self.flex_dihedral.append(flex_dihedral)
            self.molSymbol.append(molSymbol)

        atoms = copy.copy(ext.atoms)
        molCenter = atoms.get_center_of_mass(scaled=True)
        oldCell = atoms.get_cell()
        newCell = self.get_cell()
        atoms.translate(-np.dot(molCenter,oldCell))
        atoms.set_cell(newCell)
        atoms.translate(np.dot(molCenter, newCell))
        self.atoms.extend(atoms)
        self._goodBonds = {}
        self._valences = {}
        self._valenceElectrons = {}
        self._mDM = None
        self._moleculeTypesToFormula = {}

    def __add__(self, other):
        """
        Special method for supporting the '+' operator on AtomicStructure.
        Creates a copy of the first argument and extends it with the second, then returns the resulting structure.

        :type other: :class:`AtomicStructure` or descendant
        :param other: AtomicStructure to be added to the current one.
        :rtype: :class:`AtomicStructure` or descendant
        :return: resulting structure.
        """
        res = copy.copy(self)
        res.extend(other)
        return res

    def __getitem__(self, item):
        """
        Special method for retrieving an atom from structure.

        :type item: :class:`AtomicStructure` or descendant
        :param item: index of an atom to be retrieved.
        :rtype: `:class:`ase.Atom`
        :return: ase.Atom object describing an atom with given index.
        """
        return self.atoms.__getitem__(item)

    @property
    def molecules(self):
        """
        In the current implementation this method creates each molecule in list, including atomic coordinates copying,
        each time it is called. So it is relatively slow. It needs to be optimized at some point.

        :rtype: list of :class:`AtomicStructure`
        :return: list of molecules composing the current structure.
        """
        mols = []
        for inds, frmt, flex_dihedral, molSymbol in zip(self._molecules, self.format, self.flex_dihedral, self.molSymbol):
            mol = AtomicStructure(symbols=self[inds], cell=self.get_cell(), **self.config)
            if len(mol) > 1:
                mol.merge(list(range(len(mol))), frmt, flex_dihedral, molSymbol)
            mols.append(mol)
        return mols

    # TODO write Test
    def __delitem__(self, key):
        """
        Special method which deletes an atom or a group of atoms from AtomicStructure.
        The atom or group of atoms to be deleted must be a molecule. I.e. one can not delete a part of molecule.

        :type key: int, slice or list
        :param key: index, slice or list of indices determining atom or atoms to be deleted from the structure.
        """
        if isinstance(key, int):
            key = [key]
        elif isinstance(key, slice):
            key = list(range(key.start, key.stop, key.step))

        try:
            i = self._molecules.index(key)
        except ValueError:
            raise RuntimeError('Can not remove from AtomicStructure object set of atoms which is not a molecule')

        inds = list(range(len(self.chemicalSymbols)))
        inds = sorted(list(set(inds) - set(key)))

        for j in key:
            del self.atoms[j]
        del self._molecules[i]
        del self.format[i]
        del self.flex_dihedral[i]
        del self.molSymbol[i]
        for i, molecule in enumerate(self._molecules):
            self._molecules[i] = []
            for j in molecule:
                self._molecules[i].append(inds.index(j))

    def removeMolecule(self, i):
        """
        Removes a molecule from the structure.

        :type i: int
        :param i: index of the molecule to be removed.
        """
        del self[self._molecules[i]]

    def merge(self, indices: list, format: list, flex_dihedral: list, molSymbol: str):
        """
        Merges a group of atoms into a molecule.
        The molecule gets 'format' and 'flex_dihedral' information which is needed for ZMatrix construction.
        Even if an atom belongs to another molecule it can be merged.
        It is useful for future cases of tetrahedral composed structures
        where different 'molecules' can have common atoms.
        But if the new molecule is a superset of an old one, the old one will be removed.

        :type indices: list of int
        :param indices: list of indices of atoms to be merged into a molecule.
        :type format: list of tuples of int
        :param format: list of formats of atoms to be merged into a molecule (see ZMatrix definition).
        :type flex_dihedral: list of list of int
        :param flex_dihedral: list of flexible dihedral angles in the molecule (see ZMatrix definition).
        :type molSymbol: str
        :param molSymbol: name unique for this molecule's geometry.
        """
        if len(indices) != len(format):
            raise RuntimeError('Dimensions of molecule parameters do not match.')
        molecules = []
        frmts = []
        flex_dihedrals = []
        molSymbols = []
        for inds, frmt, fd, symbol in ((inds, frmt, fd, symbol) for inds, frmt, fd, symbol in
                                       zip(self._molecules, self.format, self.flex_dihedral, self.molSymbol)
                                       if not set(inds) <= set(indices)):
            molecules.append(inds)
            frmts.append(frmt)
            flex_dihedrals.append(fd)
            molSymbols.append(symbol)
        molecules.append(indices)
        frmts.append(format)
        flex_dihedrals.append(flex_dihedral)
        molSymbols.append(molSymbol)
        self._molecules = molecules
        self.format = frmts
        self.flex_dihedral = flex_dihedrals
        self.molSymbol = molSymbols

    @property
    def moleculeTypesToFormula(self):
        if not self._moleculeTypesToFormula:
            for symbol, mol in zip(self.molSymbol, self.molecules):
                if symbol not in self._moleculeTypesToFormula:
                    self._moleculeTypesToFormula[symbol] = mol.composition.elementalComposition
        return self._moleculeTypesToFormula

    def __imul__(self, m):
        """
        Special method for supporting '*=' operator. This operator creates a supercell structure of the current one.

        :type m: int or tuple
        :param m:
            Either int or tuple (int,int,int) describing supercell parameters in each direction.
            A single int parameter is treated as if a tuple of three equal parameters is given.
        """
        size = len(self.atoms)
        self.atoms.__imul__(m)
        if isinstance(m, int):
            m = (m, m, m)
        molecules = copy.copy(self._molecules)
        for i in range(1, np.product(m)):
            for j, molecule in enumerate(molecules):
                molecule = (np.asarray(molecule) + i*size).tolist()
                self._molecules.append(molecule)
                self.format.append(self.format[i])
                self.flex_dihedral.append(self.flex_dihedral[i])
                self.molSymbol.append(self.molSymbol[i])
        return self

    def __eq__(self, other):
        return cosine_distance(self.fingerprint, other.fingerprint,
                               self.fingerprintWeights, other.fingerprintWeights) < self.fingerprintTolerance

    def _calcFingerprint(self):
        Rmax = RMAX_DEFAULT
        sigma = SIGMA_DEFAULT
        delta = DELTA_DEFAULT
        if 'fingerprints' in self.config:
            fingerprintsParams = self.config['fingerprints']
            if 'Rmax' in fingerprintsParams:
                Rmax = fingerprintsParams['Rmax']
            if 'sigma' in fingerprintsParams:
                sigma = fingerprintsParams['sigma']
            if 'delta' in fingerprintsParams:
                delta = fingerprintsParams['delta']
        uniqueSimbols, inverse, numIons = np.unique(self.chemicalSymbols, return_inverse=True, return_counts=True)
        indices = np.argsort(inverse)
        revertIndices = np.argsort(indices)
        coordinates = self.scaled_coordinates[indices]
        dist_matrix = make_matrices(coordinates, self.cell, numIons, Rmax=Rmax)
        order, fing, atom_fing = fingerprint(self.volume, dist_matrix, numIons,
                                             Rmax=Rmax, sigma=sigma, delta=delta)
        self._order = order[revertIndices]
        self._fingerprint = {}
        for i, symbol1 in enumerate(uniqueSimbols):
            for j, symbol2 in enumerate(uniqueSimbols):
                self._fingerprint[(symbol1,symbol2)] = fing[i*len(uniqueSimbols) + j]
        self._atomFingerprint = []
        for i in revertIndices:
            atomFingerprint = {}
            for j, symbol in enumerate(uniqueSimbols):
                atomFingerprint[symbol] = atom_fing[i,j]
            self._atomFingerprint.append(atomFingerprint)

    @property
    def fingerprint(self):
        '''
        :rtype: Dict[Tuple[str,str], np.ndarray]
        :return: fingerprint of the structure.
        '''
        if not self._fingerprint:
            self._calcFingerprint()
        return self._fingerprint

    @property
    def atomFingerprint(self):
        '''
        :rtype: List[Dict[str], np.ndarray]]
        :return: atomic fingerprint of the structure.
        '''
        if not self._atomFingerprint:
            self._calcFingerprint()
        return self._atomFingerprint

    @property
    def order(self):
        '''
        :rtype: List[float]
        :return: local order for each atom.
        '''
        if not self._order:
            self._calcFingerprint()
        return self._order

    @property
    def averageOrder(self):
        '''
        :rtype: float
        :return: average local order for the structure.
        '''
        if np.any(np.isfinite(self.order)):
            a_order = np.mean(self.order[np.isfinite(self.order)])
        else:
            a_order = np.nan
        return a_order

    @property
    def fingerprintWeights(self):
        '''
        :rtype: Dict[Tuple[str,str], float]
        :return: weights of fingerprints of each atom type pair to be used in cosine distance calculation.
        '''
        if not self._fingerprintWeights:
            uniqueSimbols = np.unique(self.chemicalSymbols)
            weightSum = 0
            for symbol1 in uniqueSimbols:
                for symbol2 in uniqueSimbols:
                    weight = self.composition[symbol1] * self.composition[symbol2]
                    self._fingerprintWeights[(symbol1, symbol2)] = weight
                    weightSum += weight
            for key in self._fingerprintWeights.keys():
                self._fingerprintWeights[key] /= weightSum
        return self._fingerprintWeights

    @property
    def isMolecular(self):
        for mol in self._molecules:
            if len(mol) > 1: return True
        return False

    @property
    def volumeType(self):
        return self.config['volumeType'] if 'volumeType' in self.config else ('mol' if self.isMolecular else 'atom')

    @property
    def goodBonds(self):
        if not self._goodBonds:
            if 'goodBonds' in self.config:
                self._goodBonds = {tuple(x.split()) : value for x, value in self.config['goodBonds'].items()}
            else:
                self._goodBonds = {}
            goodBond = lambda symbol: Element(symbol).good_bonds
            for s1, s2 in combinations_with_replacement(np.unique(self.chemicalSymbols), r=2):
                if (s1,s2) in self._goodBonds:
                    self._goodBonds[(s2, s1)] = self._goodBonds[(s1, s2)]
                elif (s2,s1) in self._goodBonds:
                    self._goodBonds[(s1, s2)] = self._goodBonds[(s2, s1)]
                else:
                    self._goodBonds[(s1, s2)] = self._goodBonds[(s2, s1)] = np.power(goodBond(s1) * goodBond(s2), 0.5)
        return self._goodBonds

    @property
    def valences(self):
        if not self._valences:
            for symbol in np.unique(self.chemicalSymbols):
                self._valences[symbol] = Element(symbol).valence
            if 'valences' in self.config:
                self._valences.update(self.config['valences'])
        return self._valences

    @property
    def valenceElectrons(self):
        if not self._valenceElectrons:
            for symbol in np.unique(self.chemicalSymbols):
                self._valenceElectrons[symbol] = Element(symbol).valence_electrons
            if 'valenceElectrons' in self.config:
                self.valenceElectrons.update(self.config['valenceElectrons'])
        return self._valenceElectrons


    @property
    def mDM(self):
        if self._mDM is None:
            uniqueSimbols = np.unique(self.chemicalSymbols)
            if 'ionDistances' in self.config:
                minDistMatrix = {tuple(x.split()) : value for x, value in self.config['ionDistances'].items()}
            else:
                minDistMatrix = {}
            radii = {symbol: calcVolume(self.externalPressure, symbol, self.volumeType) ** (1.0 / 3.0)
                     for symbol in uniqueSimbols}
            for s1, s2 in combinations_with_replacement(uniqueSimbols, 2):
                if (s1, s2) in minDistMatrix:
                    minDistMatrix[(s2, s1)] = minDistMatrix[(s1, s2)]
                elif (s2, s1) in minDistMatrix:
                    minDistMatrix[(s1, s2)] = minDistMatrix[(s2, s1)]
                elif self.volumeType != 'mol':
                    minDistMatrix[(s1, s2)] = minDistMatrix[(s2, s1)] = min(0.22 * (radii[s1] + radii[s2]), 1.2)
                else:
                    minDistMatrix[(s1, s2)] = minDistMatrix[(s2, s1)] = 0.45 * (radii[s1] + radii[s2])

            N = len(self.atoms)
            self._mDM = np.zeros((N, N), dtype=float)
            chemicalSimbols = self.atoms.get_chemical_symbols()
            for i,j in combinations_with_replacement(range(N), 2):
                s1 = chemicalSimbols[i]
                s2 = chemicalSimbols[j]
                self._mDM[i, j] = self._mDM[j, i] = minDistMatrix[(s1, s2)]
        return self._mDM

    @property
    def composition(self) -> Composition:
        """
        :rtype: dict {str : int}
        :return: Composition of current structure in form {symbol : amount}
        """
        if len(self.molSymbol) == 1:
            count = Counter()
            for symbol in self.chemicalSymbols:
                count[symbol] += 1
            composition = Composition({self.molSymbol[0]: 1}, {self.molSymbol[0]: dict(count)})
        else:
            count = Counter()
            for symbol in self.molSymbol:
                count[symbol] += 1
            composition = Composition(count, self.moleculeTypesToFormula)
        return composition

    @property
    def chemicalSymbols(self):
        """
        :rtype: list of str
        :return: List of chemical symbols for each atom in structure
        """

        return self.atoms.get_chemical_symbols()

    @property
    def coordinates(self):
        """
        :rtype: numpy array
        :return: array of atomic absolute coordinates.
        """
        return self.atoms.get_positions()

    @property
    def scaled_coordinates(self):
        """
        :rtype: numpy array
        :return: array of atomic relative coordinates.
        """
        return self.atoms.get_scaled_positions()

    @property
    def scaled_mol_coordinates(self):
        """
        :rtype: numpy array
        :return: array of molecular centers relative coordinates.
        """
        return [molecule.get_center_of_mass(scaled=True) for molecule in self.molecules]

    @property
    def lattice(self):
        """
        :rtype: 3x3 numpy array
        :return: lattice vectors.
        """
        return self.atoms.get_cell().copy()

    @property
    def cell(self):
        """
        :rtype: 3x3 numpy array
        :return: lattice vectors.
        """
        return self.atoms.get_cell().copy()

    @property
    def cell_lengths_and_angles(self):
        """
        :rtype: numpy array
        :return: cell parameters: a, b, c, alpha, beta, gama.
        """
        return self.atoms.get_cell_lengths_and_angles()

    @property
    def volume(self):
        """
        :rtype: float
        :return: volume of the structure or the unit cell if periodic.
        """
        return self.atoms.get_volume()

    @property
    def dielectricTensor(self):
        """
        :rtype: numpy array
        :return: dielectric tensor if it is calculated, else None
        """
        return None if self._dielectricTensor is None else self._dielectricTensor.copy()

    @property
    def forces(self):
        """
        :rtype: numpy array
        :return: array of forces if it is calculated, else None
        """
        return None if self._forces is None else self._forces.copy()

    @property
    def pressureTensor(self):
        """
        :rtype: numpy array
        :return: pressure tensor if it is calculated, else None
        """
        return None if self._pressureTensor is None else self._pressureTensor.copy()

    @property
    def covalentRadii(self):
        """
        :rtype: numpy array
        :return: array of atomic covalent radii.
        """
        return np.fromiter((Element(atom).covalent_radius for atom in self.get_chemical_symbols()), dtype=float)

    @property
    def principleAxis(self):
        """
        :rtype: 3x3 numpy array
        :return: principle axes, main axes of inertia tensor (with all atom masses set to be equal).
        """
        coordinates = self.coordinates - self.coordinates.mean(axis=0)
        inertia = np.zeros((3, 3), dtype=float)  # moment of inertia tensor
        inertia[0, 0] = (coordinates[:, 1] ** 2 + coordinates[:, 2] ** 2).sum()
        inertia[1, 1] = (coordinates[:, 0] ** 2 + coordinates[:, 2] ** 2).sum()
        inertia[2, 2] = (coordinates[:, 0] ** 2 + coordinates[:, 1] ** 2).sum()
        inertia[0, 1] = -(coordinates[:, 0] * coordinates[:, 1]).sum()
        inertia[1, 2] = -(coordinates[:, 1] * coordinates[:, 2]).sum()
        inertia[2, 0] = -(coordinates[:, 2] * coordinates[:, 0]).sum()
        inertia[1, 0] = -(coordinates[:, 0] * coordinates[:, 1]).sum()
        inertia[2, 1] = -(coordinates[:, 1] * coordinates[:, 2]).sum()
        inertia[0, 2] = -(coordinates[:, 2] * coordinates[:, 0]).sum()
        return np.linalg.eigh(inertia)

    def _molecule_CN(self, i: int):
        """
        Method which roughly (very roughly!!!) estimates the coordination numbers of a molecule.

        :type i: int
        :param i: Molecule index.
        :rtype: numpy array
        :return: Array of coordination numbers.
        """
        molecule = self.molecules[i]
        radiu = np.array([Element(atom).covalent_radius for atom in molecule.get_chemical_symbols()])
        CN = np.fromiter((len(neighbours) for neighbours in find_pair(molecule.coordinates, radiu)), dtype=int)
        return CN

    def rotatePrinciple(self, i: int, axis: int, angle: float):
        """
        Rotate the *i* th molecule with respect to *axis* =[0,1,2] principle axis by the angle *angle* .

        :type i: int
        :param i: index of molecule to rotate.
        :type axis: int
        :param axis: index of axis [0,1,2] with respect to which the molecule should be rotated.
        :type angle: float
        :param angle: angle by which the molecule should be rotated.
        """
        assert axis < 3
        molecule = self.molecules[i]
        molecule.set_masses([1] * len(molecule))
        values, vectors = molecule.get_moments_of_inertia(vectors=True)
        ref = np.max(values)
        value = values[axis]
        vector = vectors[axis]
        if value > 0.0001:
            angle *= ref/value
            molecule.rotate(vector, angle)
        positions = self.atoms.arrays.get('positions')
        assert len(positions[self._molecules[i]]) == len(molecule.get_positions())
        positions[self._molecules[i]] = molecule.get_positions()

    def rotateFlexDiherdal(self, i: int, j: int, angle: float):
        """
        Rotate the *j* th flexible dihedral angle of *i* th molecule by the angle *angle* .

        :type i: int
        :param i: index of molecule to rotate.
        :type j: int
        :param j: index of flexible dihedral angle to rotate.
        :type angle: float
        :param angle: angle by which the dihedral should be rotated.
        """
        assert j < len(self.flex_dihedral[i])
        molecule = self.molecules[i]
        zmatrix = coord2Zmatrix(molecule.coordinates, np.asarray(self.format[i], dtype=int))
        molecule_mod = copy.deepcopy(molecule)
        zmatrix[self.flex_dihedral[i][j], 2] += angle
        molecule_mod.set_positions(zmatrix2coord(zmatrix, np.asarray(self.format[i], dtype=int)))
        if not np.array_equal(molecule._molecule_CN(0), molecule_mod._molecule_CN(0)):
            raise RuntimeError("CN mismatch")
        molecule.set_positions(molecule_mod.get_positions())
        positions = self.atoms.arrays.get('positions')
        assert len(positions[self._molecules[i]]) == len(molecule.get_positions())
        positions[self._molecules[i]] = molecule.get_positions()

    def optimizeLattice(self):
        """
        Optimize structure lattice in case of ill formed (too prolongated) structures.
        """
        coor = self.get_positions()
        lat = self.get_cell()
        coor, lat = optLattice(coor, lat)
        lat = AtomicStructure(cell = lat).get_cell_lengths_and_angles()
        self.set_cell(lat)
        self.set_positions(coor)

    def set_cell(self, cell: np.ndarray, optimize: bool=False, **kwargs):
        """
        Set up lattice parameters for the structure and calls optimizeLattice if needed.

        :type cell: 3x3 or 6 numpy array
        :param cell: lattice vectors or cell parameters to be set for the structure.
        :type optimize: bool
        :param optimize: If True we will call :meth:`optimizeLattice` right after setting. If False we wont.
        :type kwargs: dict
        :param kwargs: additional arguments and keywords to be passed to :meth:`ase.Atoms.set_cell`.
        """
        self.atoms.set_cell(cell, **kwargs)
        if optimize:
            self.optimizeLattice()

    def __copy__(self):
        """
        Special method which allows correctly make a copy of current structure using 'copy()' operator.

        :rtype: :class:`AtomicStructure` or descendant
        :return: A copy of the structure with conserving type of it.
        """
        return self.fromDICT(self.toDICT())

    def translate_scaled(self, displacement: np.ndarray):
        """
        Translate the structure by a given displacement relative to the unit cell.

        :type displacement: numpy array
        :param displacement: array of scaled distances.
        """
        self.atoms.translate(np.dot(displacement, self.atoms.cell))

    def isGoodDistances(self, symbols: list = None, minDistMatrix: np.ndarray = None) -> bool:
        """
        Check if the structure meets the minimal distance constraints provided with Minimal Distances Matrix.

        :type symbols: list of str
        :param symbols: list of chemical symbols used in minimal distances matrix.
        :type minDistMatrix: square numpy array
        :param minDistMatrix: minimal distances matrix.
        :rtype: bool
        :return: True if the structure meet the constraint, False otherwise.
        """
        if len(self.atoms) < 2:
            return True
        actualDistances = self.get_all_distances(mic=np.any(self.atoms.get_pbc()))
        constNeighbours = np.vstack([np.eye(3), -np.eye(3)])
        for inds, molecule in zip(self._molecules, self.molecules):
            distVectorsMatrix = molecule.get_all_distances(mic=True, vector=True)
            for i, distVectorsRow in enumerate(distVectorsMatrix):
                for j, vect in enumerate(distVectorsRow):
                    vect = self.cell.scaled_positions(vect)
                    if np.all(np.abs(vect) < 1.0):
                        dists = np.linalg.norm(vect + constNeighbours, axis=1)
                        distVectorsMatrix[i,j] = self.cell.cartesian_positions(vect + constNeighbours[np.argmin(dists)])
            actualDistances[tuple(np.meshgrid(inds, inds))] = np.linalg.norm(distVectorsMatrix, axis=2)
        return np.all(actualDistances >= self.mDM.T)

    def isMoleculesDistinct(self) -> bool:
        """
        Check if molecules do not interpenetrate each other.
        The algorithm is as follows. For each atom, distances to every other atom are calculated
        and then normalized over the sum of covalent radii. The atom with shortest such distance
        should be in the same molecule as the first one.

        :rtype: bool
        :return: True if check passed, False otherwise.
        """
        actualMinDistances = self.get_all_distances(mic=np.any(self.atoms.get_pbc()))
        noPbcMinDistances = self.get_all_distances(mic=False)
        normalizedMinDistances = actualMinDistances / (self.covalentRadii.reshape(-1, 1) +
                                                       self.covalentRadii.reshape(1, -1))
        for inds in self._molecules:
            if len(inds) > 1:
                for i in inds:
                    j = np.argsort(normalizedMinDistances[i])[1]
                    if (j not in inds) or (not np.isclose(actualMinDistances[i, j], noPbcMinDistances[i, j])):
                        return False
        return True

    def isGoodSystem(self):
        return self.isGoodDistances()

    def toDICT(self) -> dict:
        """
        Create a dictionary representation of the structure.

        :rtype: dict
        :return: Dictionary representing the structure.
        """
        dct = super().toDICT()
        dct['symbols'] = self.atoms.get_chemical_symbols()
        dct['molecules'] = self._molecules
        del dct['_molecules']
        dct['molSymbols'] = self.molSymbol
        del dct['molSymbol']
        dct['molFormats'] = self.format
        del dct['format']
        dct['molFlexDihedrals'] = self.flex_dihedral
        del dct['flex_dihedral']
        dct['cell'] = self.atoms.get_cell().tolist()
        dct['positions'] = self.atoms.get_positions().tolist()
        dct['pbc'] = self.get_pbc().tolist()
        charges = self.atoms.get_initial_charges()
        if np.any(charges):
            dct['charges'] = charges.tolist()
        del dct['atoms']
        if self._dielectricTensor is not None:
            dct['dielectricTensor'] = self._dielectricTensor.tolist()
            del dct['_dielectricTensor']
        if self._forces is not None:
            dct['forces'] = self._forces.tolist()
        if self._pressureTensor is not None:
            dct['pressureTensor'] = self._pressureTensor.tolist()
            del dct['_pressureTensor']

        del dct['_goodBonds']
        del dct['_valences']
        del dct['_valenceElectrons']
        del dct['_mDM']

        # TODO refactor this
        if 'bonds' in dct:
            del dct['bonds']
        if 'f' in dct:
            del dct['f']
        if 'softmodes' in dct:
            del dct['softmodes']
        if 'lastUsedModeIter' in dct:
            del dct['lastUsedModeIter']
        return dct

    @classmethod
    def fromDICT(cls, dct: dict):
        """
        Reconstruct :class:`AtomicStructure` from its dictionary representation.

        :type dct: dict
        :param dct: Dictionary representing the structure.
        :rtype: type corresponding to this classmethod. (see Examples)
        :return: new created structure.
        """
        dct = copy.copy(dct)
        newStructure = cls(symbols=dct['symbols'])
        del dct['symbols']
        if 'cell' in dct:
            newStructure.set_cell(dct['cell'])
            del dct['cell']
        newStructure.set_positions(dct['positions'])
        del dct['positions']
        newStructure.set_pbc(dct['pbc'])
        del dct['pbc']
        assert len(dct['molecules']) == len(dct['molFormats']) and \
               len(dct['molecules']) == len(dct['molFlexDihedrals']) and \
               len(dct['molecules']) == len(dct['molSymbols']), \
            f'Molecule specification mismatch:' \
            f' molecules, formats, flex_dihedrals, symbols:' \
            f' {len(dct["molecules"])}, {len(dct["molFormats"])},' \
            f' {len (dct["molFlexDihedrals"])}, {len(dct["molSymbols"])}.'
        for molecule, frmt, flex_dihedral, molSymbol in zip(dct['molecules'], dct['molFormats'],
                                                            dct['molFlexDihedrals'], dct['molSymbols']):
            newStructure.merge(molecule, frmt, flex_dihedral, molSymbol)
        del dct['molecules']
        del dct['molFormats']
        del dct['molFlexDihedrals']
        del dct['molSymbols']
        if 'charges' in dct:
            newStructure.set_initial_charges(dct['charges'])
            del dct['charges']
        if 'dielectricTensor' in dct:
            newStructure._dielectricTensor = dct['dielectricTensor']
            del dct['dielectricTensor']
        if 'pressureTensor' in dct:
            newStructure._pressureTensor = np.asarray(dct['pressureTensor'])
            del dct['pressureTensor']
        if 'forces' in dct:
            newStructure._forces = np.asarray(dct['forces'])
            del dct['forces']
        dct['_goodBonds'] = {}
        dct['_valences'] = {}
        dct['_valenceElectrons'] = {}
        dct['_mDM'] = None

        newStructure.__dict__.update(dct)
        return newStructure

    @System.isBad.setter
    def isBad(self, value : bool):
        System.isBad.fset(self, value)
        if value:
            self.enthalpy = np.inf
