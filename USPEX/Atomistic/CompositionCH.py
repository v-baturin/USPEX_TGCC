from USPEX.Expressions.ConvexHull import ConvexHull
from USPEX.Expressions.ExpressionEvaluator import ExpressionEvaluator
from USPEX.Expressions.Functions.BasicFunctions import BasicFunctions
from .CompositionSpace import CompositionSpace


class CompositionCH(ConvexHull):
    def __init__(self, systems: list, compositionSpace: CompositionSpace):
        self.systems = systems
        self.compositionSpace = compositionSpace
        extensions = dict(
            basic=BasicFunctions(),
            compositionSpace=compositionSpace.expressionExtension(compositionSpace),
        )
        expression = ('getRelativeCHSpace',
                      ('compositionSpace.numBlocksFromCompositions', 'simpleMoleculeUtility.composition.origin'),
                      '.enthalpy.origin'
                      )
        super().__init__(ExpressionEvaluator(self.systems, extensions).evaluate(expression))

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
        extensions = dict(
            basic=BasicFunctions(),
            compositionSpace=self.compositionSpace.expressionExtension(self.compositionSpace),
        )
        expression = ('getRelativeCHSpace',
                      ('compositionSpace.numBlocksFromCompositions', 'simpleMoleculeUtility.composition.origin'),
                      '.enthalpy.origin'
                      )
        super().__init__(ExpressionEvaluator(self.systems, extensions).evaluate(expression))
