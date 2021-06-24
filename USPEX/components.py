from .Atomistic.Element import Element
from .Atomistic.CellUtility import Cell
from .Atomistic.AtomicPrimitives import AtomicStructure, AtomicDisassembler

from .IO.CrystalRepresentation import CrystalRepresentation
CrystalRepresentation.registerTypes(AtomicStructure, Element, Cell, AtomicDisassembler)

from .Target import Target
from .Atomistic.CompositionSpace import CompositionSpace
from .Atomistic.RadialDistributionUtility import RadialDistributionUtility
from .Atomistic.CellUtility import CellUtility
from .Atomistic.SimpleMoleculeUtility import SimpleMoleculeUtility
SimpleMoleculeUtility.registerTypes(AtomicStructure, Element, Cell, AtomicDisassembler)
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
Seeds.registerTypes(CrystalRepresentation)
from .VariationOperators import VariationOperators
variationOperators = VariationOperators(hybridizationTypes=[Heredity],
                                        mutationTypes=[Softmodemutation, Permutation, Transmutation],
                                        creationTypes=[RandTop, RandSym],
                                        seedsType=Seeds)
Target.registerTarget('Crystal', [CompositionSpace, RadialDistributionUtility, CellUtility, SimpleMoleculeUtility,
                                  Conditions, IonDistances, PowderSpectrumAnalyzer, SingleCrystalSpectrumAnalyzer], variationOperators)

from .GlobalOptimizer import GlobalOptimizer
from .Fitness import Fitness
from .Selection.USPEXClassic import USPEXClassic
GlobalOptimizer.setFitnessType(Fitness)
GlobalOptimizer.registerSelection(USPEXClassic)

from .GenerationController import GenerationController
GenerationController.registerOptimizer(GlobalOptimizer)

from .Calculators.ABINIT_Interface import ABINIT_Interface
ABINIT_Interface.registerTypes(AtomicStructure, Element, Cell, AtomicDisassembler)

from .Calculators.GULP_Interface import GULP_Interface
GULP_Interface.registerTypes(AtomicStructure, Element, Cell, AtomicDisassembler)

from .Calculators.LAMMPS_Interface import LAMMPS_Interface
LAMMPS_Interface.registerTypes(AtomicStructure, Element, Cell, AtomicDisassembler)

from .Calculators.MLIP_Interface import MLIP_Interface
MLIP_Interface.registerTypes(AtomicStructure, Element, Cell, AtomicDisassembler)

from .Calculators.PWmat_Interface import PWmat_Interface
PWmat_Interface.registerTypes(AtomicStructure, Element, Cell, AtomicDisassembler)

from .Calculators.QE_Interface import QE_Interface
QE_Interface.registerTypes(AtomicStructure, Element, Cell, AtomicDisassembler)

from .Calculators.VASP_Interface import VASP_Interface
VASP_Interface.registerTypes(AtomicStructure, Element, Cell, AtomicDisassembler)
