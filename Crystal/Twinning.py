import logging
logger = logging.getLogger(__name__)


'''
@file        Twinning.py
@author:     Evgeny Tikhonov
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    e.tikhonov@physics.msu.ru
@date        2018/01/31
@brief       Twinning variational operator, improved
-B -m cProfile -o output.prof
'''

import numpy as np
from copy import copy
from time import time

from ..VarOperator import VarOperator, VOFailed
from ..Atomistic.Element import Element

class TwinningException(Exception):
    pass

class Twinning(VarOperator):
    def __init__(self, systemFactory, config, pool, utilities,
                 mode_weights : tuple=(1.5, 1.0, 0.5), axis_weights : tuple=(1.0, 1.0),
                 glide_weights : tuple=(1.0/6.0,1.0/6.0,1.0/3.0,1.0/3.0)):
        super(Twinning, self).__init__(systemFactory, config, pool, utilities)
        self.compositionSpace = utilities['compositionSpace']
        self.MAX_ATTEMPTS = 100
        self.MAX_TIME = 60
        self.correlation_coefficient = None
        self.mode_weights = np.array(mode_weights) / np.sum(mode_weights)
        self.axis_weights = np.array(axis_weights) / np.sum(axis_weights)
        self.glide_weights = np.array(glide_weights) / np.sum(glide_weights)
        self.modes = []
        self.twin_plane_direction = None
        self.offspring = None
        self.used_shifts = {'mirroring': [],
                           'axis': [],
                           'inversion': []}  # change to dictionary with mode parameters!
        self.shift = None
        self.parameters = {}
        if self.compositionSpace.isFixedComposition:
            self.desiredComposition = dict(zip(self.compositionSpace.symbols, self.compositionSpace.blocks[0]))
        else:
            self.desiredComposition = None
            # TODO: composition must respect blocks anyway
        self.mismatched_composition = {}
        self.Rz = np.array([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0,-1.0]])
        self.I = np.array([[-1.0, 0.0, 0.0],
                           [ 0.0,-1.0, 0.0],
                           [ 0.0, 0.0,-1.0]])
        self.axis_order = n = 2.0  # n-axis in future # TODO
        self.Cnz = ([[np.cos(2.0 * np.pi / n), np.sin(2.0 * np.pi / n), 0.0],
                    [-np.sin(2.0 * np.pi / n), np.cos(2.0 * np.pi / n), 0.0],
                     [0.0, 0.0, 1.0]])
        self.start_time = None

    def tune(self, population : list):
        order = []
        enthalpy = []
        for system in population:
            order.append(system.averageOrder)
            enthalpy.append(system.enthalpy)
        self.correlation_coefficient = np.corrcoef(order, enthalpy)[0, 1]
        if np.isnan(self.correlation_coefficient):
            self.correlation_coefficient = None

    def __call__(self, parent):

        # Step 1. Select twinning mode: mirroring, rotation axis or inversion
        logger.debug(f"Structure {parent.ID} is going to be a parent")
        self.modes = []
        self.used_shifts = {'mirroring': [],
                            'axis': [],
                            'inversion': []}  # change to dictionary with mode parameters!
        for i in range(self.MAX_ATTEMPTS):
            self.modes.append(np.random.choice(['mirroring', 'axis', 'inversion'], p=self.mode_weights))
        # self.logger.debug('TWINNING: {}'.format(''.join([mode[0] for mode in self.modes])))
        for attempt, mode in enumerate(self.modes):  # Main loop
            try:
                self.parameters['mode'] = mode
                self.offspring = copy(parent)
                # First, self.offspring is used to select twinning direction and change all coordinates
                # Then, self.offspring is emptied and future images are put into it
                # Step 2. Select direction in cell for mirror plane
                self.twin_plane_direction = np.random.choice(['x', 'y', 'z'])
                scaled_coords = self.offspring.scaled_coordinates
                if self.twin_plane_direction == 'x':
                    scaled_coords[:, [0, 2]] = scaled_coords[:, [2, 0]]
                    self.offspring.set_positions(np.dot(scaled_coords, self.offspring.cell))
                elif self.twin_plane_direction == 'y':
                    scaled_coords[:, [0, 1]] = scaled_coords[:, [1, 0]]
                    self.offspring.set_positions(np.dot(scaled_coords, self.offspring.cell))
                elif self.twin_plane_direction == 'z':
                    pass
                else:
                    raise RuntimeError
                if mode in ['mirroring','axis']:
                    # Orthogonalize cell
                    lattice = self.offspring.cell
                    lattice[2][0:2] = 0.0, 0.0
                    self.offspring.set_cell(lattice)
                self.molecules = self.offspring.molecules
                self.offspring = []
                # There we prepare self.offspring to accept child images
                shift = self.pick_shift()
                # In case of axis rotation, we need to rotate around some arbitrary point, not around cell center
                axis_shift = np.random.random(3)
                axis_shift[2] = 0.0
                self.bad_structure = []
                for molecule in self.molecules:
                    self.molecule = molecule
                    # This is to randomize position of mirroring plane
                    self.shift = shift
                    self.molecule.translate_scaled(-shift)
                    molecule.set_cell(parent.get_cell())
                    offset = np.floor(molecule.get_center_of_mass(scaled=True))
                    self.molecule.translate_scaled(-offset)
                    # self.offspring.wrap()
                    # This is to move reflected atoms below Z = 0, above Z = 0 will be image
                    self.molecule.translate_scaled(np.array([0.0, 0.0, -0.5]))
                    if mode == 'mirroring':

                        image = copy(self.molecule)
                        child = self.molecule + image
                        child_positions = np.copy(image.positions)
                        image.set_positions(np.dot(image.positions, self.Rz))
                        glide = np.random.choice(['a', 'b', 'n', 'm'], p=self.glide_weights)
                        if glide == 'a':
                            # Glide reflection image along x
                            image.set_positions(np.dot(image.get_scaled_positions() + np.array([0.5, 0.0, 0.0]), image.cell))
                        elif glide == 'b':
                            # Glide reflection image along y
                            image.set_positions(np.dot(image.get_scaled_positions() + np.array([0.0, 0.5, 0.0]), image.cell))
                        elif glide == 'n':
                            # Glide reflection image along x + y
                            image.set_positions(np.dot(image.get_scaled_positions() + np.array([0.5, 0.5, 0.0]), image.cell))
                        elif glide == 'm':
                            pass
                        else:
                            raise RuntimeError
                        child_positions = np.concatenate((child_positions, image.positions), axis=0)
                        child.set_positions(child_positions)
                        if self.molecule.get_center_of_mass(scaled=True)[2] >= 0:
                            self.bad_structure.extend(child.molecules)
                        else:
                            self.offspring.extend(child.molecules)

                    elif mode == 'axis':
                        axis = np.random.choice(['2', '21'], p=self.axis_weights)
                        self.molecule.translate_scaled(axis_shift)
                        if axis == '2':
                            self.parameters['axis_glide'] = False
                            # Rotation twinning. Choose one of two diagonals for rotation:
                            image = copy(self.molecule)
                            child = self.molecule + image
                            child_positions = np.copy(image.positions)
                            image.set_positions(np.dot(image.positions, self.Cnz))
                            child_positions = np.concatenate((child_positions, image.positions), axis=0)
                            child.set_positions(child_positions)
                            triangle = np.random.choice(['a+b', 'a-b'])
                            if triangle == 'a+b':
                                if self.molecule.get_center_of_mass(scaled=True)[0] + \
                                        self.molecule.get_center_of_mass(scaled=True)[1] >= 1.0:
                                    self.bad_structure.extend(child.molecules)
                                else:
                                    self.offspring.extend(child.molecules)
                            elif triangle == 'a-b':
                                if self.molecule.get_center_of_mass(scaled=True)[0] - \
                                        self.molecule.get_center_of_mass(scaled=True)[1] >= 0.0:
                                    self.bad_structure.extend(child.molecules)
                                else:
                                    self.offspring.extend(child.molecules)
                            else:
                                raise RuntimeError


                        elif axis == '21':
                            self.parameters['axis_glide'] = True
                            # Rotation twinning with glide

                            image = copy(self.molecule)
                            child = self.molecule + image
                            child_positions = np.copy(image.positions)
                            image.set_positions(np.dot(image.positions, self.Cnz))
                            image.set_positions(np.dot(image.get_scaled_positions() + np.array([0.0, 0.0, 1.0 / self.axis_order]), image.cell))
                            child_positions = np.concatenate((child_positions, image.positions), axis=0)
                            child.set_positions(child_positions)
                            if self.molecule.get_center_of_mass(scaled=True)[2] >= 0.0:
                                self.bad_structure.extend(child.molecules)
                            else:
                                self.offspring.extend(child.molecules)
                        else:
                            raise RuntimeError
                    elif mode == 'inversion':
                        # Inversion twinning
                        self.molecule.translate_scaled(np.array([-0.5, -0.5, 0.0]))
                        if self.molecule.get_center_of_mass(scaled=True)[2] >= 0:
                            continue
                        image = copy(self.molecule)
                        child = self.molecule + image
                        child_positions = np.copy(image.positions)
                        child_positions = np.concatenate((child_positions, np.dot(image.positions, self.I)), axis=0)
                        child.set_positions(child_positions)
                        if self.molecule.get_center_of_mass(scaled=True)[2] >= 0:
                            self.bad_structure.extend(child.molecules)
                        else:
                            self.offspring.extend(child.molecules)
                    else:
                        raise RuntimeError
                # Remove too close atoms
                # self.offspring.wrap()
                # self.remove_too_close() TODO fix issue with too_close_pairs not knowing eath other ([2,3] [3,4] etc)
                for molecule in self.offspring:
                    molecule.translate_scaled(-np.floor(molecule.get_center_of_mass(scaled=True)))
                for molecule in self.bad_structure:
                    molecule.translate_scaled(-np.floor(molecule.get_center_of_mass(scaled=True)))
                self.offspring = self.systemFactory(molecules=self.offspring, cell=parent.cell, **self.config)
                if self.desiredComposition is not None:
                    self.remove_extra_atoms()
                    self.add_missing_atoms()

                if self.distance_check() and  self.compositionSpace.isGoodComposition(self.offspring.composition):
                    if parent == self.offspring:
                        logger.debug('Twinning: child coincidences with parent!')
                        if attempt == self.MAX_ATTEMPTS - 1:
                            raise VOFailed
                        else:
                            continue
                    final_structure = self.offspring
                    self.pool.assignID(final_structure)
                    final_structure.howCome = self.__class__.__name__
                    final_structure.parent = str(parent.ID)
                    logger.info(f"Structure {final_structure.ID} created via {mode} from {parent.ID} parent")
                    return (final_structure,)
                else:
                    logger.debug(f"Failed generate structure via {mode} from {parent.ID} parent")
            except TwinningException:
                logger.debug(f"Failed generate structure via {mode} from {parent.ID} parent")
                continue

        raise VOFailed

    def remove_too_close(self):
        # Merge too close atoms into one
        try:
            D = self.offspring.get_all_distances(mic=any(self.offspring.atoms.get_pbc()))
        except ValueError:
            return
        DM = np.zeros_like(D)
        too_close_pairs = []
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                if i > j:
                    DM[i][j] = Element(self.offspring[i].symbol).covalent_radius + Element(self.offspring[j].symbol).covalent_radius
                    if D[i][j] < DM[i][j] * 0.1:
                        too_close_pairs.append((i, j))

        index_to_pop = []
        index_to_merge = []
        positions_to_merge = []
        for i, j in too_close_pairs:
            assert i > j
            index_to_pop.append(i)
            index_to_merge.append(j)
            merged_position = 0.5 * (self.offspring.positions[i] + self.offspring.positions[j])
            positions_to_merge.append(merged_position)

        for index, position in zip(index_to_merge, positions_to_merge):
            self.offspring[index].position = position

        for index in reversed(sorted(index_to_pop)):
            #self.offspring.pop(index)
            try:
                self.offspring.removeMolecule(index)
            except RuntimeError:
                raise TwinningException

    def check_composition(self):
        # makes dictionary where we have extra atoms (+) or lack them(-)
        s, c = np.unique(self.offspring.molSymbol, return_counts=True)
        s = s.tolist()
        c = c.tolist()
        composition = dict(zip(s, c))
        for element in self.desiredComposition.keys():
            if element not in composition.keys():
                composition.update({element: 0})
            new_value = {element : composition[element] - self.desiredComposition[element]}
            self.mismatched_composition.update(new_value)
        mismatch = []
        symbols = []
        for Z in self.desiredComposition.keys():
            mismatch.append(self.mismatched_composition[Z])
            symbols.append(Z)
        return symbols, np.array(mismatch)

    def remove_extra_atoms(self):
        symbols, mismatch = self.check_composition()
        new_offspring = []
        for s, m in zip(symbols, mismatch):
            candidate = [molecule for molecule in self.offspring.molecules if molecule.molSymbol[0] == s][max(-m, 0):]
            # max(-m,0) handle case when we lack atoms and don't need to remove, but still need to save them
            # We need to remove m molecules, so we remove m molecules from end
            # Todo: if remove by ordering in future, it may start from [abs(m):] - rethink index there
            new_offspring.extend(candidate)
        self.offspring = self.systemFactory(molecules=new_offspring, cell=self.offspring.cell, **self.config)

    def find_pair(self, ifr, index_for_replace):
        # return None  # TODO WTF who break my twinning?
        first = self.offspring.get_scaled_positions()[ifr]
        if self.parameters['mode'] == 'mirroring':
            second = np.array([first[0], first[1], 1 - first[2]])
        elif self.parameters['mode'] == 'axis':
            if self.parameters['axis_glide']:
                second = np.array([1 - first[0], 1 - first[1], 1 - first[2]])
            else:
                second = np.array([1 - first[0], 1 - first[1], first[2]])
        elif self.parameters['mode'] == 'inversion':
            second = np.array([-first[0], -first[1], -first[2]])
        else:
            raise RuntimeError
        found_second = None
        for i in index_for_replace:
            if np.allclose(self.offspring[i].position, second):
                found_second = i
                logger.debug('Twinning: pair found')
                break
        return found_second

    def add_missing_atoms(self):
        """
        Adding atoms should respect twinning mode
        If number of atoms is odd, we first try add it to center and center plane
        :return:
        """
        start_time = time()
        MAX_TRY_ADD = 100
        symbols, mismatch = self.check_composition()
        for s, m in zip(symbols, mismatch):
            for candidate in self.bad_structure:
                if candidate.molSymbol[0] == s:
                    if m == 0:
                        break
                    self.offspring.extend(candidate)
                    m += 1

    def distance_check(self, with_atoms=None):
        if with_atoms is not None:
            symbols = with_atoms['symbols']
            new_pos = with_atoms['new_pos']
            if new_pos.shape == (3,):
                added = self.systemFactory(symbols=symbols, positions=new_pos.reshape((1, 3)), cell=self.offspring.cell,
                                           **self.config)
                new_pos = new_pos.reshape((1, 3))
            else:
                added = self.systemFactory(symbols=symbols * new_pos.shape[0], positions=new_pos, cell=self.offspring.cell,
                                           **self.config)
            check_me = self.offspring.atoms + added.atoms
            check_me_positions = np.concatenate((self.offspring.positions, new_pos), axis=0)
            check_me_crystal = self.systemFactory(symbols=check_me.get_chemical_symbols(), positions=check_me_positions,
                                                  cell=self.offspring.cell, **self.config)
            return check_me_crystal.isGoodDistances()
        else:
            try:
                return self.offspring.isGoodSystem()
            except ValueError:
                return False

    def pick_shift(self):
        if self.correlation_coefficient is not None:  # correlation coefficient shift goes there
            shift = None  # TODO Rewrite pick_shift function
            mode = self.parameters['mode']
            shift = np.array([0.0, 0.0, np.random.random()])
            isNewFound = False
            if not len(self.used_shifts[mode]):
                isNewFound = True
            else:
                tries = 20
                while tries > 0:
                    tries -= 1
                    if not np.isclose(np.array(self.used_shifts[mode]), shift)[:, 2].any():
                        isNewFound = True
                        break
        else:
            mode = self.parameters['mode']
            shift = np.array([0.0, 0.0, np.random.random()])
            isNewFound = False
            if not len(self.used_shifts[mode]):
                isNewFound = True
            else:
                tries = 20
                while tries > 0:
                    tries -= 1
                    if not np.isclose(np.array(self.used_shifts[mode]), shift)[:,2].any():
                        isNewFound = True
                        break
        if isNewFound:
            self.used_shifts[mode].append(shift)
        return shift
