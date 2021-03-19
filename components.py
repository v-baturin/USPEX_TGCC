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
from .Atomistic.Operators.RandSym import RandSym
from .Atomistic.Operators.Softmodemutation import Softmodemutation
from .VariationOperators import VariationOperators
variationOperators = VariationOperators(hybridizationTypes=[Heredity],
                                        mutationTypes=[Softmodemutation],
                                        creationTypes=[RandTop, RandSym])
Target.registerTarget('Crystal', [CompositionSpace, RadialDistributionUtility, CellUtility, SimpleMoleculeUtility,
                                  Conditions, IonDistances], variationOperators)

from .GlobalOptimizer import GlobalOptimizer
from .Fitness import Fitness
from .Selection.USPEXClassic import USPEXClassic
GlobalOptimizer.setFitnessType(Fitness)
GlobalOptimizer.registerSelection(USPEXClassic)

from .Controllers.GenerationController import GenerationController
GenerationController.registerOptimizer(GlobalOptimizer)

from .Atomistic.Element import Element
from .Atomistic.CellUtility import Cell
from .Atomistic.AtomicPrimitives import AtomicStructure, AtomicDisassembler
from .IO.Crystal.CrystalSystemRepresentation import CrystalSystemRepresentation
CrystalSystemRepresentation.registerTypes(AtomicStructure, Element, Cell, AtomicDisassembler)
from .Calculators.GULP_Interface import GULP_Interface
GULP_Interface.registerTypes(AtomicStructure, Element, Cell, AtomicDisassembler)

from .InputParser import read
from .IO.compileParams import compileParams
