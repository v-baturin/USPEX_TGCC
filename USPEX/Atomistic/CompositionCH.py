from types import SimpleNamespace

from ..Fitness.Private.ConvexHull import ConvexHull
from ..Fitness.ExpressionEvaluator import ExpressionEvaluator
from ..Fitness.BasicFunctions import BasicFunctions
from ..Optimizers.SystemPool import SystemPool
from .CompositionSpace import CompositionSpace


class CompositionCH(ConvexHull):
    def __init__(self, systems: list, compositionSpace: CompositionSpace, simpleMoleculeUtility):
        self.systems = systems
        pool = SystemPool()
        pool.update(self.systems)
        self.compositionSpace = compositionSpace
        self.simpleMoleculeUtility = simpleMoleculeUtility
        extensions = dict(
            basic=BasicFunctions(),
            compositionSpace=compositionSpace.fitnessExtension(compositionSpace),
            simpleMoleculeUtility=simpleMoleculeUtility.fitnessExtension(simpleMoleculeUtility)
        )
        super().__init__(ExpressionEvaluator(pool.uniqueSystems, extensions).evaluate(('getRelativeCHSpace',
                                                                                      ('compositionSpace.numBlocksFromCompositions',
                                                              'simpleMoleculeUtility.composition'), 'enthalpy')))

    @property
    def lower_bound(self):
        return [self.systems[i] for i in super(CompositionCH,self).lower_bound]

    @property
    def upper_bound(self):
        return [self.systems[i] for i in super(CompositionCH,self).upper_bound]

    @property
    def depth(self):
        return super().depth

    @property
    def height(self):
        return super().height

    def extend(self, systems: list):
        self.systems.extend(systems)
        pool = SystemPool()
        pool.update(self.systems)
        extensions = dict(
            basic=BasicFunctions(),
            compositionSpace=self.compositionSpace.fitnessExtension(self.compositionSpace),
            simpleMoleculeUtility=self.simpleMoleculeUtility.fitnessExtension(self.simpleMoleculeUtility)
        )

        super().__init__(ExpressionEvaluator(pool.uniqueSystems, extensions).evaluate(('getRelativeCHSpace',
                                                                                      ('compositionSpace.numBlocksFromCompositions',
                                                              'simpleMoleculeUtility.composition'), 'enthalpy')))
