from ..VariationOperators import VariationOperators
from .RandTop import RandTop
from .RandSym import RandSym
from .Heredity import Heredity
from .Twinning import Twinning
from .Softmodemutation import Softmodemutation
from .Permutation import Permutation
from .Transmutation import Transmutation
from .Seeds import Seeds
variationOperators = VariationOperators(hybridizationTypes = [Heredity],
                                        mutationTypes = [Twinning, Softmodemutation, Permutation, Transmutation],
                                        creationTypes = [RandTop, RandSym],
                                        seedsType = Seeds)
