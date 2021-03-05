from .Target import Target
from .Atomistic.CompositionSpace import CompositionSpace
from .Atomistic.RadialDistributionUtility import RadialDistributionUtility
from .Atomistic.CellUtility import CellUtility
from .Atomistic.SimpleMoleculeUtility import SimpleMoleculeUtility
from .Atomistic.Conditions import Conditions
from .Atomistic.IonDistances import IonDistances
# from .XRay.SpectrumAnalyzer import SpectrumAnalyzer
from .Atomistic.Operators.Heredity import Heredity
from .Atomistic.Operators.RandTop import RandTop
from .Atomistic.Operators.Softmodemutation import Softmodemutation
from .VariationOperators import VariationOperators
variationOperators = VariationOperators(hybridizationTypes=[Heredity],
                                        mutationTypes=[Softmodemutation],
                                        creationTypes=[RandTop])
Target.registerTarget('Crystal', [CompositionSpace, RadialDistributionUtility, CellUtility, SimpleMoleculeUtility,
                                  Conditions, IonDistances], variationOperators)

from .GlobalOptimizer import GlobalOptimizer
from .Fitness import Fitness
from .Selection.USPEXClassic import USPEXClassic
GlobalOptimizer.setFitnessType(Fitness)
GlobalOptimizer.registerSelection(USPEXClassic)

from .Controllers.GenerationController import GenerationController
GenerationController.registerOptimizer(GlobalOptimizer)

from .InputParser import read
from .IO.compileParams import compileParams
