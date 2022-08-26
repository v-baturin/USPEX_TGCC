from os.path import join, expanduser, exists, dirname
from os import makedirs

from .IO.InputParser import read


FILENAME = join(expanduser('~'), '.config/uspex/presets.uspex')
makedirs(dirname(FILENAME), exist_ok=True)


if not exists(FILENAME):
    definitions = {
        'presetFitness': {
            'enthalpyCCH': ('convexHullHeight', ('getRelativeCHSpace', ('compositionSpace.numBlocksFromCompositions',
                                                                        'simpleMoleculeUtility.composition'), 'enthalpy')),
            'enthalpyCS': ('simpleHeight', ('getRelativeCHSpace', ('compositionSpace.numBlocksFromCompositions',
                                                                        'simpleMoleculeUtility.composition'), 'enthalpy')),
            'refinedEnergy': ('minus', 'energy', 'environmentEnergy'),
            'refinedEnergyCCH': ('convexHullHeight', ('getRelativeCHSpace', ('compositionSpace.numBlocksFromCompositions',
                                                                        'simpleMoleculeUtility.composition'), 'refinedEnergy')),
            'refinedEnthalpy': ('minus', 'enthalpy', 'environmentEnthalpy'),
            'refinedEnthalpyCCH': ('convexHullHeight', ('getRelativeCHSpace', ('compositionSpace.numBlocksFromCompositions',
                                                                         'simpleMoleculeUtility.composition'), 'refinedEnthalpy')),
            'normRefinedEnthalpy': ('divide', 'refinedEnthalpy', 'cellUtility.area'),
            'normRefinedAbsCompCH': ('convexHullHeight', ('getAbsoluteCHSpace', ('compositionSpace.numBlocksFromCompositions',
                                                                        'simpleMoleculeUtility.composition'), 'normRefinedEnthalpy')),
        },

        'presetOutput': {
            'CrystalFixComp': {
                'columns': [
                    ('simpleMoleculeUtility.composition', 'Composition'),
                    ('enthalpy', 'Enthalpy (eV)'),
                    ('cellUtility.volume', 'Volume (A^3)'),
                    ('cellUtility.symmetry', 'SYMMETRY (N)'),
                    ('radialDistributionUtility.structureOrder', 'Structure order'),
                    ('radialDistributionUtility.averageOrder', 'Average order'),
                    ('radialDistributionUtility.quasientropy', 'Quasientropy')
                ],
                'toDraw': [('dep', 'enthalpy', 'raw', 'ID', 'raw'),
                           ('dep', 'enthalpy', 'per_atom', 'ID', 'raw'),
                           ('dep', 'enthalpy', 'per_atom', 'cellUtility.volume', 'per_atom'),
                           ('stat', 'enthalpy', 'per_atom', '', '')],
                'presentConvexHull': False
            },
            'CrystalVarComp': {
                'columns': [
                    ('simpleMoleculeUtility.composition', 'Composition'),
                    ('enthalpy', 'Enthalpy (eV)'),
                    ('enthalpyCCH', 'Enthalpy per Block above CCH (eV)'),
                    ('cellUtility.volume', 'Volume (A^3)'),
                    ('cellUtility.symmetry', 'SYMMETRY (N)'),
                    ('radialDistributionUtility.structureOrder', 'Structure order'),
                    ('radialDistributionUtility.averageOrder', 'Average order'),
                    ('radialDistributionUtility.quasientropy', 'Quasientropy')
                ],
                'toDraw': [('dep', 'enthalpy', 'per_atom', 'ID', 'raw'),
                           ('dep', 'enthalpy', 'per_atom', 'cellUtility.volume', 'per_atom'),
                           ('stat', 'enthalpy', 'per_atom', '', '')],
                'presentConvexHull': True
            },
            'Nano2DFixComp': {
                'columns': [
                    ('simpleMoleculeUtility.composition', 'Composition'),
                    ('enthalpy', 'Enthalpy (eV)'),
                    ('cellUtility.area', 'Area (A^2)'),
                    ('radialDistributionUtility.structureOrder', 'Structure order'),
                    ('radialDistributionUtility.averageOrder', 'Average order'),
                    ('radialDistributionUtility.quasientropy', 'Quasientropy')
                ],
                'toDraw': [('dep', 'enthalpy', 'raw', 'ID', 'raw'),
                           ('dep', 'enthalpy', 'per_atom', 'ID', 'raw'),
                           ('stat', 'enthalpy', 'per_atom', '', '')],
                'presentConvexHull': False
            },
            'Nano2DVarComp': {
                'columns': [
                    ('simpleMoleculeUtility.composition', 'Composition'),
                    ('enthalpy', 'Enthalpy (eV)'),
                    ('enthalpyCCH', 'Enthalpy per Block above CCH (eV)'),
                    ('cellUtility.area', 'Volume (A^2)'),
                    # ('normRefinedEnthalpy', 'Formation energy per unit area (eV / A^2)'),
                    # ('normRefinedAbsCompCH', 'Formation energy per unit area above CCH (eV / A^2)'),
                    ('radialDistributionUtility.structureOrder', 'Structure order'),
                    ('radialDistributionUtility.averageOrder', 'Average order'),
                    ('radialDistributionUtility.quasientropy', 'Quasientropy')
                ],
                'toDraw': [('dep', 'enthalpy', 'per_atom', 'ID', 'raw'),
                           ('stat', 'enthalpy', 'per_atom', '', '')],
                'presentConvexHull': True
            },
            'Nano1DFixComp': {
                'columns': [
                    ('simpleMoleculeUtility.composition', 'Composition'),
                    ('enthalpy', 'Enthalpy (eV)'),
                    ('cellUtility.length', 'Period (A)'),
                    ('radialDistributionUtility.structureOrder', 'Structure order'),
                    ('radialDistributionUtility.averageOrder', 'Average order'),
                    ('radialDistributionUtility.quasientropy', 'Quasientropy')
                ],
                'toDraw': [('dep', 'enthalpy', 'raw', 'ID', 'raw'),
                           ('dep', 'enthalpy', 'per_atom', 'ID', 'raw'),
                           ('stat', 'enthalpy', 'per_atom', '', '')],
                'presentConvexHull': False
            },
            'Nano1DVarComp': {
                'columns': [
                    ('simpleMoleculeUtility.composition', 'Composition'),
                    ('enthalpy', 'Enthalpy (eV)'),
                    ('enthalpyCCH', 'Enthalpy per Block above CCH (eV)'),
                    ('cellUtility.length', 'Period (A)'),
                    ('radialDistributionUtility.structureOrder', 'Structure order'),
                    ('radialDistributionUtility.averageOrder', 'Average order'),
                    ('radialDistributionUtility.quasientropy', 'Quasientropy')
                ],
                'toDraw': [('dep', 'enthalpy', 'per_atom', 'ID', 'raw'),
                           ('stat', 'enthalpy', 'per_atom', '', '')],
                'presentConvexHull': True
            },
            'Nano0DFixComp': {
                'columns': [
                    ('simpleMoleculeUtility.composition', 'Composition'),
                    ('enthalpy', 'Enthalpy (eV)'),
                    ('radialDistributionUtility.structureOrder', 'Structure order'),
                    ('radialDistributionUtility.averageOrder', 'Average order'),
                    ('radialDistributionUtility.quasientropy', 'Quasientropy')
                ],
                'toDraw': [('dep', 'enthalpy', 'raw', 'ID', 'raw'),
                           ('dep', 'enthalpy', 'per_atom', 'ID', 'raw'),
                           ('stat', 'enthalpy', 'per_atom', '', '')],
                'presentConvexHull': False
            },
            'Nano0DVarComp': {
                'columns': [
                    ('simpleMoleculeUtility.composition', 'Composition'),
                    ('enthalpy', 'Enthalpy (eV)'),
                    ('enthalpyCS', 'Enthalpy per Block above the best for composition(eV'),
                    ('radialDistributionUtility.structureOrder', 'Structure order'),
                    ('radialDistributionUtility.averageOrder', 'Average order'),
                    ('radialDistributionUtility.quasientropy', 'Quasientropy')
                ],
                'toDraw': [('dep', 'enthalpy', 'per_atom', 'ID', 'raw'),
                           ('stat', 'enthalpy', 'per_atom', '', '')],
                'presentConvexHull': False
            }

        }
    }

else:
    definitions = read(FILENAME)

presetFitness = definitions['presetFitness']
presetOutput = definitions['presetOutput']