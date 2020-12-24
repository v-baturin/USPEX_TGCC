from typing import List

from ..ConvexHull import ConvexHull
from ..Fitness import Fitness
from ..SystemPool import SystemPool
from .CompositionSpace import CompositionSpace


class CompositionCH(ConvexHull):
    def __init__(self, systems: list, comositionSpace: CompositionSpace):
        self.systems = systems
        pool = SystemPool()
        pool.update(self.systems)
        self.compositionSpace = comositionSpace
        super().__init__(Fitness(pool, {'compositionSpace': self.compositionSpace}).calcFitness(('getRelativeCHSpace',
                                                                                                 'compositionSpace.numBlocks',
                                                                                                 'enthalpy')))

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
        super().__init__(Fitness(pool, {'compositionSpace': self.compositionSpace}).calcFitness(('getRelativeCHSpace',
                                                                                                 'compositionSpace.numBlocks',
                                                                                                 'enthalpy')))
