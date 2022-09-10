"""
USPEX.Atomistic.EnvironmentBuilder
==================================
"""

import logging
import numpy as np

from pymatgen.analysis.gb.grain import GrainBoundaryGenerator
from pymatgen.core.surface import SlabGenerator
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from USPEX.Atomistic.CellUtility import Cell
from USPEX.Atomistic.AtomicPrimitives import AtomicStructure
from USPEX.Atomistic.Element import Element
from USPEX.Atomistic.EnvironmentUtility import adjustStructures

logger = logging.getLogger(__name__)

DEFAULT_MAX_MISFIT_STRAIN = 5E-3
DEFAULT_MAX_ENVIRONMENT_AREA = 1000

def buildInterface(lowerFile, upperFile, pbc, slabThickness, adjust=False, sigma=None, plane=None, rotAxis=None, 
                    lowerPlane=None, upperPlane=None, maxMisfitStrain=None, maxEnvironmentArea=None, **kwagrs):
    if lowerFile == upperFile and sigma is not None:
        logger.debug(f'Proceeding with Grain Boundary mode')
        initStructure = EnvironmentBuilder.structureRepresentation.readAtomicStructureRaw(lowerFile, pbc=(1,1,1))
        lowerStructure, upperStructure = constructGrainsSlabs(initStructure, pbc, sigma, plane, rotAxis, slabThickness)
        logger.debug('Grains are successfully created')
    else:
        logger.debug(f'Proceeding with Heterostructure mode')
        initLowerStructure = EnvironmentBuilder.structureRepresentation.readAtomicStructureRaw(lowerFile, pbc=(1,1,1))
        initUpperStructure = EnvironmentBuilder.structureRepresentation.readAtomicStructureRaw(upperFile, pbc=(1,1,1))
        lowerStructure = constructSurfaceSlab(initLowerStructure, pbc, lowerPlane, slabThickness)
        upperStructure = constructSurfaceSlab(initUpperStructure, pbc, upperPlane, slabThickness)
        logger.debug('Surface Slabs are successfully created')
    if adjust:
        logger.debug(f'Auto-adjustment of interfacial slabs was enabled')
        antiPBC = tuple((~np.asarray(pbc, dtype=bool)).tolist())
        nonPBCAxis = np.flatnonzero(antiPBC)[0]
        maxMisfitStrain = maxMisfitStrain if maxMisfitStrain else DEFAULT_MAX_MISFIT_STRAIN
        maxEnvironmentArea = maxEnvironmentArea if maxEnvironmentArea else DEFAULT_MAX_ENVIRONMENT_AREA
        lowerStructure, upperStructure, _ = adjustStructures(lowerStructure, upperStructure, axis=nonPBCAxis, 
                                                                maxMisfitStrain=maxMisfitStrain,
                                                                maxSubstrateArea=maxEnvironmentArea)
        logger.debug(f'Interfacial slabs are successfully adjusted')
    environment = dict(
        lowerStructure=lowerStructure,
        upperStructure=upperStructure,
    )
    return environment

def buildSubstrate(file, pbc, plane, slabThickness, **kwargs):
    initStructure = EnvironmentBuilder.structureRepresentation.readAtomicStructureRaw(file)
    structure = constructSurfaceSlab(initStructure, pbc, plane, slabThickness)
    environment = dict(
        structure=structure,
    )
    return environment

# TODO.
# 1. Implement build via compileParams
# 2. Make sure the rest of the parameters (gap, bufferThickness) are properly gathered
# 3. Test new Environment Utility


class EnvironmentBuilder:
    """
    Class building environment objects for a given description.
    """
    structureRepresentation = None
    supportedBuilders = {
        'interface': buildInterface,
        'substrate': buildSubstrate
    }

    @classmethod
    def setRepresentation(cls, representation):
        cls.structureRepresentation = representation

    @staticmethod
    def build(environments: list):
        for environment in environments:
            build = environment.pop('build')
            if build:
                envType = environment.get('type').lower()
                logger.debug(f'Selected environment type is {envType}')
                builderMethod = EnvironmentBuilder.supportedBuilders.get(envType)
                if builderMethod is not None:
                    #TODO Clear environment dict from all builder settings
                    environment.update(**builderMethod(**environment))
                else:
                    raise ValueError(f"Unknown environment type {envType}.")
        return environments
        

def convertFromPymatgen(pmgStructure, pbc):
    """
    Converts the pymatgen Structure object to the USPEX AtomicStructure object
    """
    cellVectors = pmgStructure.lattice.matrix[np.flatnonzero(pbc)]
    cell = Cell.initFromCellVectors(pbc, cellVectors)
    species = [Element(specie.name) for specie in pmgStructure.species]
    coordinates = pmgStructure.cart_coords
    structure = AtomicStructure(species, coordinates, cell)
    return structure

def convertToPymatgen(structure):
    """
    Converts the USPEX AtomicStructure object to the pymatgen Structure object 
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
