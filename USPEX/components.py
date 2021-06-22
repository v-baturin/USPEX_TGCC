from .Target import Target
from .Atomistic.CompositionSpace import CompositionSpace
from .Atomistic.RadialDistributionUtility import RadialDistributionUtility
from .Atomistic.CellUtility import CellUtility
from .Atomistic.SimpleMoleculeUtility import SimpleMoleculeUtility
from .Atomistic.Conditions import Conditions
from .Atomistic.IonDistances import IonDistances
from .XRay.PowderSpectrumAnalyzer import PowderSpectrumAnalyzer
from .XRay.SingleCrystalSpectrumAnalyzer import SingleCrystalSpectrumAnalyzer
from .Atomistic.Operators.Heredity import Heredity
from .Atomistic.Operators.RandTop import RandTop
from .Atomistic.Operators.RandSym import RandSym
from .Atomistic.Operators.Softmodemutation import Softmodemutation
from .Atomistic.Operators.Permutation import Permutation
from .Atomistic.Operators.Transmutation import Transmutation
from .Atomistic.Operators.Seeds import Seeds
Target.registerTarget('Crystal',
                      utilities=[CompositionSpace, RadialDistributionUtility, CellUtility, SimpleMoleculeUtility,
                                 Conditions, IonDistances, PowderSpectrumAnalyzer, SingleCrystalSpectrumAnalyzer],
                      hybridizations=[Heredity],
                      mutations=[Softmodemutation, Permutation, Transmutation],
                      creations=[RandTop, RandSym],
                      seeds=Seeds)

from .GlobalOptimizer import GlobalOptimizer
from .Fitness import Fitness
from .Selection.USPEXClassic import USPEXClassic
GlobalOptimizer.setFitnessType(Fitness)
GlobalOptimizer.registerSelection(USPEXClassic)

from .GenerationController import GenerationController
GenerationController.registerOptimizer(GlobalOptimizer)

from .Atomistic.Element import Element
from .Atomistic.CellUtility import Cell
from .Atomistic.AtomicPrimitives import AtomicStructure, AtomicDisassembler

SimpleMoleculeUtility.registerTypes(AtomicStructure, Element, Cell, AtomicDisassembler)
from .IO.CrystalRepresentation import CrystalRepresentation
CrystalRepresentation.registerTypes(AtomicStructure, Element, Cell, AtomicDisassembler)
from .Calculators.GULP_Interface import GULP_Interface
GULP_Interface.registerTypes(AtomicStructure, Element, Cell, AtomicDisassembler)

Seeds.registerTypes(CrystalRepresentation)
