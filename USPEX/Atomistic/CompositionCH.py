from USPEX.Expressions.ConvexHull import ConvexHull
from USPEX.Expressions.ExpressionEvaluator import ExpressionEvaluator
from USPEX.Expressions.Functions.BasicFunctions import BasicFunctions
from ..Optimizers.SystemPool import SystemPool
from .CompositionSpace import CompositionSpace


class CompositionCH(ConvexHull):
    def __init__(self, systems: list, compositionSpace: CompositionSpace):
        self.systems = systems
        pool = SystemPool()
        pool.update(self.systems)
        self.compositionSpace = compositionSpace
        extensions = dict(
            basic=BasicFunctions(),
            compositionSpace=compositionSpace.expressionExtension(compositionSpace),
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
            compositionSpace=self.compositionSpace.expressionExtension(self.compositionSpace),
        )

        super().__init__(ExpressionEvaluator(pool.uniqueSystems, extensions).evaluate(('getRelativeCHSpace',
                                                                                      ('compositionSpace.numBlocksFromCompositions',
                                                              'simpleMoleculeUtility.composition'), 'enthalpy')))
