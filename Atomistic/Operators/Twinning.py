import numpy as np

from ..Slab import Slab


class Twinning:

    def __init__(self, cellUtility, compositionSpace, radialDistributionUtility):
        self.cellUtility = cellUtility
        self.compositionSpace = compositionSpace
        self.radialDistributionUtility = radialDistributionUtility
        self.correlation = 0

    def __call__(self, system, *args, **kwargs):
        cell = system['cell']
        molecules = system['molecules']
        order = self.radialDistributionUtility.order(system)

        axis = np.random.randint(3)
        mode = np.random.choice(['mirroring', 'axis', 'inversion'], p=self.mode_weights)

        slab11, slab12 = Slab.getRandomSlabs(molecules=molecules, inputCell=cell, outputCell=cell,
                                             axis=axis, gaugesOfSlabs=(1, 1),
                                             order=order, correlation=self.correlation, parity=0)

        moleculesTransformed = transform(molecules, cell, mode, axis)

        goodCandidateMolecules = molecules[slab11.indices] + moleculesTransformed[slab11.indices]
        goodCandidateOrder = slab11.depths + slab11.depths
        goodCandidateMolecules = goodCandidateMolecules[np.argsort(goodCandidateOrder)]

        badCandidateMolecules = moleculesTransformed[slab12.indices] + molecules[slab12.indices]
        badCandidateOrder = slab12.depths + slab12.depths
        badCandidateMolecules = badCandidateMolecules[np.argsort(badCandidateOrder)]

        outputCell = cell # TODO

        composition = self.compositionSpace.calculateComposition(molecules)
        desiredComposition = self.compositionSpace.findDesiredComposition(composition, composition, self.compositionSpace.calculateComposition(goodCandidateMolecules))[0]

        goodCandidateMolecules = removeExtra(goodCandidateMolecules, self.compositionSpace, desiredComposition)
        goodCandidateMolecules += addLacking(badCandidateMolecules,
                                             desiredComposition - self.compositionSpace.calculateComposition(goodCandidateMolecules))

        return ({'molecules' : goodCandidateMolecules, 'cell': outputCell},)


def removeExtra(molecules, compositionSpace, desiredComposition):
    """
    remove extra molecules from child_good structure
    :param molecules:
    :param compositionSpace:
    :param desiredComposition:
    :return:
    """
    moleculesNew = []
    for molecule in molecules:
        if compositionSpace.calculateComposition(moleculesNew)[molecule] < desiredComposition[molecule]:
            moleculesNew.append(molecule)
    return moleculesNew


def addLacking(molecules, desiredComposition):
    """
    add lacking molecules to good_child from bad_child
    :param molecules:
    :param desiredComposition:
    :return:
    """
    moleculesNew = []
    for symbol, desired in desiredComposition.items():
        moleculesIter = iter(molecules)
        while desired > 0:
            molecule = next(moleculesIter)
            if molecule.symbol == symbol:
                moleculesNew.append(molecule)
                desired -= 1
    return moleculesNew

def someLogic(self, mode):
    
    offspring = copy(parent)
    # First, offspring is used to select twinning direction and change all coordinates
    # Then, offspring is emptied and future images are put into it
    # Step 2. Select direction in cell for mirror plane
    twin_plane_direction = np.random.choice(['x', 'y', 'z'])
    scaled_coords = offspring.scaled_coordinates
    if twin_plane_direction == 'x':
        scaled_coords[:, [0, 2]] = scaled_coords[:, [2, 0]]
        offspring.set_positions(np.dot(scaled_coords, offspring.cell))
    elif twin_plane_direction == 'y':
        scaled_coords[:, [0, 1]] = scaled_coords[:, [1, 0]]
        offspring.set_positions(np.dot(scaled_coords, offspring.cell))
    elif twin_plane_direction == 'z':
        pass
    else:
        raise RuntimeError
    if mode in ['mirroring', 'axis']:
        # Orthogonalize cell
        lattice = offspring.cell
        if twin_plane_direction == 'x':
            if not np.isclose(lattice[0, 0], 0, atol=0.5):
                lattice[0][1] = 0.0
                lattice[0][2] = 0.0
        elif twin_plane_direction == 'y':
            if not np.isclose(lattice[1, 1], 0, atol=0.5):
                lattice[1][0] = 0.0
                lattice[1][2] = 0.0
        elif twin_plane_direction == 'z':
            if not np.isclose(lattice[2, 2], 0, atol=0.5):
                lattice[2][0] = 0.0
                lattice[2][1] = 0.0
        offspring.set_cell(lattice)
    molecules = offspring.molecules
    offspring = []
    # There we prepare offspring to accept child images
    shift = self.pick_shift()
    # In case of axis rotation, we need to rotate around some arbitrary point, not around cell center
    axis_shift = np.random.random(3)
    axis_shift[2] = 0.0
    badStructure = []
    for molecule in molecules:
        self.molecule = molecule
        # This is to randomize position of mirroring plane
        self.molecule.translate_scaled(-shift)
        molecule.set_cell(parent.get_cell())
        offset = np.floor(molecule.get_center_of_mass(scaled=True))
        self.molecule.translate_scaled(-offset)
        # offspring.wrap()
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
                badStructure.extend(child.molecules)
            else:
                offspring.extend(child.molecules)
    
        elif mode == 'axis':
            axis = np.random.choice(['2', '21'], p=self.axis_weights)
            self.molecule.translate_scaled(axis_shift)
            if axis == '2':
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
                        badStructure.extend(child.molecules)
                    else:
                        offspring.extend(child.molecules)
                elif triangle == 'a-b':
                    if self.molecule.get_center_of_mass(scaled=True)[0] - \
                            self.molecule.get_center_of_mass(scaled=True)[1] >= 0.0:
                        badStructure.extend(child.molecules)
                    else:
                        offspring.extend(child.molecules)
                else:
                    raise RuntimeError
    
    
            elif axis == '21':
                # Rotation twinning with glide
    
                image = copy(self.molecule)
                child = self.molecule + image
                child_positions = np.copy(image.positions)
                image.set_positions(np.dot(image.positions, self.Cnz))
                image.set_positions(
                    np.dot(image.get_scaled_positions() + np.array([0.0, 0.0, 1.0 / self.axis_order]), image.cell))
                child_positions = np.concatenate((child_positions, image.positions), axis=0)
                child.set_positions(child_positions)
                if self.molecule.get_center_of_mass(scaled=True)[2] >= 0.0:
                    badStructure.extend(child.molecules)
                else:
                    offspring.extend(child.molecules)
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
                badStructure.extend(child.molecules)
            else:
                offspring.extend(child.molecules)
