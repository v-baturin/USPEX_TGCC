import numpy as np
import logging

from pymatgen.analysis.interfaces.zsl import ZSLGenerator
from pymatgen.analysis.interfaces.coherent_interfaces import get_2d_transform, Deformation

from pymatgen.analysis.gb.grain import GrainBoundaryGenerator
from pymatgen.core.surface import SlabGenerator
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


logger = logging.getLogger(__name__)


# TODO create a unittest for all environment types
def adjustSystem(molecules, cell, envStructure, axis, maxSubstrateArea, maxMisfitStrain, returnSupercellMatrices=False):
    """
    Adjusts the cells of the environment and the structure to make them fit each other
    """
    logger.debug(f'Got the system with {len(molecules)} atoms')
    logger.debug(f'Got the envStructure with {len(envStructure)} atoms')
    structure, disassembler = EnvironmentUtility.structureType.assemble(molecules, cell)
    newEnvStructure, newStructure, supercellMatrices = adjustStructures(envStructure, structure, axis=axis,
                                                    maxMisfitStrain=maxMisfitStrain, maxSubstrateArea=maxSubstrateArea)
    disassembler = type(disassembler)(indices=[[i] for i in range(len(newStructure))], cell=newStructure.getCell(), environment=None)
    newSystem = disassembler.disassemble(newStructure)
    newMolecules, newCell = newSystem['molecules'], newSystem['cell']
    logger.debug(f'Got the final system with {len(newMolecules)} atoms')
    logger.debug(f'Got the final newEnvStructure with {len(newEnvStructure)} atoms')
    if returnSupercellMatrices:
        return newMolecules, newCell, newEnvStructure, supercellMatrices
    else:
        return newMolecules, newCell, newEnvStructure

def adjustStructures(lowerStructure, upperStructure, axis, maxSubstrateArea, maxMisfitStrain):
    """
    Adjusts unit cells of two structures along the given axis to make them fit each other
    """
    lowerCellVectors = lowerStructure.getCell().getCellVectors()
    upperCellVectors = upperStructure.getCell().getCellVectors()
    lowerSupercellMatrix, upperSupercellMatrix = calculateSupercellMatrices(lowerCellVectors, upperCellVectors, axis=axis,
                                                                     maxMisfitStrain=maxMisfitStrain, maxSubstrateArea=maxSubstrateArea)
    logger.debug(f'Supercell Matrices: {lowerSupercellMatrix} (lower), {upperSupercellMatrix} (upper)')
    newLowerStructure = lowerStructure.makeSupercell(lowerSupercellMatrix)
    newUpperStructure = upperStructure.makeSupercell(upperSupercellMatrix)
    newUpperStructure = alignStructure(newUpperStructure, newLowerStructure)
    assert len(newLowerStructure) % len(lowerStructure) == 0
    assert len(newUpperStructure) % len(upperStructure) == 0
    supercellMatrices = (lowerSupercellMatrix, upperSupercellMatrix)
    return newLowerStructure, newUpperStructure, supercellMatrices

def calculateSupercellMatrices(filmCellVectors, substrateCellVectors, axis, maxMisfitStrain, maxSubstrateArea):
    """
    Calculates 2D supercell matrices with non-periodic specified axis for both film and substrate
    to make them fit each other in terms of the minimal resulting supercell misfit strain
    """
    logger.debug('Starting the calculation of supercell matrices')
    generator = ZSLGenerator(max_area=maxSubstrateArea)
    logger.debug('ZSLGenerator initialized')
    filmPlaneCellVectors = np.delete(filmCellVectors, axis, axis=0)
    substratePlaneCellVectors = np.delete(substrateCellVectors, axis, axis=0)
    matches = list(generator(filmPlaneCellVectors, substratePlaneCellVectors))
    logger.debug(f'Found {len(matches)} matches for current maxArea of {maxSubstrateArea}')
    if len(matches) > 0:
        metrics, matrices = [], []
        for match in matches:
            M1 = np.round(get_2d_transform(filmPlaneCellVectors, match.film_sl_vectors)).astype(int)
            M2 = np.round(get_2d_transform(substratePlaneCellVectors, match.substrate_sl_vectors)).astype(int)
            strain = Deformation(match.match_transformation).green_lagrange_strain
            metrics.append([np.max(strain), match.match_area])
            matrices.append([M1, M2])
        metrics = np.array(metrics)
        matrices = np.array(matrices)
        if len(matrices) > 0:
            goodStrainIndices = np.where(metrics[:, 0] <= maxMisfitStrain)[0]
            logger.debug(f'{len(goodStrainIndices)} matches satisfy maxMisfitStrain of {maxMisfitStrain:.3e} ')
            if len(goodStrainIndices) > 0:
                goodMatricesIndex = metrics[goodStrainIndices][:, 1].argmin()
                goodMetrics = metrics[goodStrainIndices][goodMatricesIndex]
                M1, M2 = matrices[goodStrainIndices][goodMatricesIndex]
                filmSupercellMatrix = convert2DMatrixTo3D(M1, axis=axis)
                substrateSupercellMatrix = convert2DMatrixTo3D(M2, axis=axis)
                logger.debug(f'Successfully found supercell matrices')
                logger.debug(f'Misfit strain: {goodMetrics[0]:.4e}, Area: {int(goodMetrics[1])}')
                logger.debug('Calculation of supercell matrices is finished')
                return filmSupercellMatrix, substrateSupercellMatrix
    else:
        raise RuntimeError('ZSLGenerator failed.')

def alignStructure(structure, targetStructure):
    """
    Transforms the cellVectors of a given structure to map them closely to
    cellVectors of the targetStructure.
    """
    cell = structure.getCell()
    targetCell = targetStructure.getCell()
    newCell = cell.getOrthogonallyTransformedCell(targetCell)
    newStructure = EnvironmentUtility.structureType.initFromFractionalCoordinates(structure.getAtomTypes(),
                                                                structure.getFractionalCoordinates(), newCell)
    return newStructure

def convert2DMatrixTo3D(matrix, axis):
    """
    Expands 2D matrix to 3D by filling the diagonal element on the specified axis with 1.0
    and offdiagonal elements with zeros
    """
    M = np.diag((1.0, 1.0, 1.0))
    rows = np.delete(np.arange(3), axis)
    M[np.ix_(rows, rows)] = matrix
    return M

def convert3DMatrixTo2D(matrix, axis):
    """
    Crops 3D matrix and makes 2D one by deleting the row and column corresponding to the specified axis
    """
    rows = np.delete(np.arange(3), axis)
    return matrix[np.ix_(rows, rows)]

def convertFromPymatgen(pmgStructure, pbc):
    """
    Converts the pymatgen Structure object to the USPEX EnvironmentUtility.structureType object
    """
    cellVectors = pmgStructure.lattice.matrix[np.flatnonzero(pbc)]
    cell = EnvironmentUtility.cellType.initFromCellVectors(pbc, cellVectors)
    species = [EnvironmentUtility.atomType(specie.name) for specie in pmgStructure.species]
    coordinates = pmgStructure.cart_coords
    structure = EnvironmentUtility.structureType(species, coordinates, cell)
    return structure

def convertToPymatgen(structure):
    """
    Converts the USPEX EnvironmentUtility.structureType object to the pymatgen Structure object
    """
    cellVectors = structure.getRectifiedCell().getCellVectors()
    coordinates = structure.getCartesianCoordinates()
    species = [el.short_name for el in structure.getAtomTypes()]
    pmgStructure = Structure(cellVectors, species, coordinates, coords_are_cartesian=True)
    return pmgStructure

def constructSurfaceSlab(structure, pbc, plane, slabThickness, **kwargs):
    """
    Creates a surface slab with given plane Miller indices and slabThickness
    """
    logger.debug(f'Surface Slab Constructor is initialized')
    pmgStructure = convertToPymatgen(structure)
    logger.debug('Creating slab with parameters:')
    logger.debug(f'Plane: {plane}')
    slabGenerator = SlabGenerator(pmgStructure, miller_index=plane, min_slab_size=slabThickness,
                                  min_vacuum_size=1e-3, lll_reduce=True, center_slab=True, **kwargs)
    slab = slabGenerator.get_slab()
    slabStructure = convertFromPymatgen(slab, pbc)
    return slabStructure

def constructGrainsSlabs(structure, pbc, sigma, plane, rotAxis, slabThickness, **kwargs):
    """
    Creates two grain slabs with given plane Miller indices, Sigma value and rotation axis
    within CSL Model.
    """
    logger.debug(f'Grains Constructor is initialized')
    antiPBC = tuple((~np.asarray(pbc, dtype=bool)).tolist())
    nonPBCAxis = np.flatnonzero(antiPBC)[0]
    pmgStructure = convertToPymatgen(structure)
    spgAnalzyer = SpacegroupAnalyzer(pmgStructure)
    crystalSystem = spgAnalzyer.get_crystal_system()
    logger.debug(f'Grains crystal system was determined as {crystalSystem}')
    gbGenerator = GrainBoundaryGenerator(initial_structure=pmgStructure)
    angle = min(gbGenerator.get_rotation_angle_from_sigma(sigma, rotAxis, lat_type=crystalSystem[0]))
    logger.debug('Creating graines with parameters:')
    logger.debug(f'Sigma: {sigma}, Angle: {angle:.3f}, Plane: {plane}, Axis: {rotAxis}')
    gb = gbGenerator.gb_from_parameters(rotAxis, angle, plane=plane, expand_times=1, rm_ratio=0.5)
    unitSlabThickness = np.linalg.norm(gb.lattice.matrix[nonPBCAxis]) / 2
    expandTimes = int(np.ceil(slabThickness / unitSlabThickness))
    gb = gbGenerator.gb_from_parameters(rotAxis, angle, plane=plane, expand_times=expandTimes, rm_ratio=0.5)
    assert None not in gb.site_properties['grain_label']
    lowerGrain = SpacegroupAnalyzer(gb.bottom_grain).get_refined_structure()
    upperGrain = SpacegroupAnalyzer(gb.top_grain).get_refined_structure()
    grainThickness = lowerGrain.cart_coords[:, nonPBCAxis].max() - lowerGrain.cart_coords[:, nonPBCAxis].min()
    upperShiftVector = np.zeros(3)
    lowerShiftVector = np.zeros(3)
    for i in range(3):
        if i == nonPBCAxis:
            upperShiftVector[i] = grainThickness / 2 + 1e-5
            lowerShiftVector[i] = - lowerGrain.cart_coords[:, i].min()
    upperGrain.translate_sites(indices=list(range(len(upperGrain))), vector=upperShiftVector, frac_coords=False)
    lowerGrain.translate_sites(indices=list(range(len(lowerGrain))), vector=lowerShiftVector, frac_coords=False)
    lowerGrainStructure = convertFromPymatgen(lowerGrain, pbc)
    upperGrainStructure = convertFromPymatgen(upperGrain, pbc)
    lowerGrainStructure = centerAndEnvelope(lowerGrainStructure)
    upperGrainStructure = centerAndEnvelope(upperGrainStructure)
    return lowerGrainStructure, upperGrainStructure

def centerAndEnvelope(structure):
    cell = structure.getRectifiedCell()
    coordinates = structure.getCartesianCoordinates()
    coordinates = cell.center(coordinates)
    structure._cell = cell
    structure._coordinates = coordinates
    return structure
