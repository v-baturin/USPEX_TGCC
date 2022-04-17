from .Atomistic.Element import Element
from .Atomistic.CellUtility import Cell
from .Atomistic.AtomicPrimitives import AtomicStructure, AtomicDisassembler

from .IO.AtomisticRepresentation import AtomisticRepresentation
AtomisticRepresentation.registerTypes(AtomicStructure, Element, Cell, AtomicDisassembler)

from .GlobalOptimizer import GlobalOptimizer
from .Fitness import Fitness
GlobalOptimizer.setFitnessType(Fitness)
from .Selection.USPEXClassic import USPEXClassic
GlobalOptimizer.registerSelection(USPEXClassic)
from .Atomistic.CompositionSpace import CompositionSpace
from .Atomistic.EnvironmentUtility import EnvironmentUtility
EnvironmentUtility.setRepresentation(AtomisticRepresentation)
from .Atomistic.RadialDistributionUtility import RadialDistributionUtility
from .Atomistic.CellUtility import CellUtility
from .Atomistic.SimpleMoleculeUtility import SimpleMoleculeUtility
SimpleMoleculeUtility.registerTypes(AtomicStructure, Element, Cell, AtomicDisassembler)
from .Atomistic.Conditions import Conditions
from .Atomistic.IonDistances import IonDistances
from .Atomistic.Constraints import Constraints
from .XRay.PowderSpectrumAnalyzer import PowderSpectrumAnalyzer
from .XRay.SingleCrystalSpectrumAnalyzer import SingleCrystalSpectrumAnalyzer
from .Atomistic.Operators.Heredity import Heredity
from .Atomistic.Operators.RandTop import RandTop
from .Atomistic.Operators.RandSym import RandSym
from .Atomistic.Operators.RandSymPyXtal import RandSymPyXtal
from .Atomistic.Operators.Softmodemutation import Softmodemutation
from .Atomistic.Operators.Permutation import Permutation
from .Atomistic.Operators.Transmutation import Transmutation
from .Atomistic.Operators.Seeds import Seeds
Seeds.registerTypes(AtomisticRepresentation)
GlobalOptimizer.registerTarget('Atomistic',
                      utilities=[CompositionSpace, RadialDistributionUtility, CellUtility, EnvironmentUtility, SimpleMoleculeUtility,
                                 Conditions, IonDistances, Constraints,
                                 PowderSpectrumAnalyzer, SingleCrystalSpectrumAnalyzer],
                      hybridizations=[Heredity],
                      mutations=[Softmodemutation, Permutation, Transmutation],
                      creations=[RandTop, RandSym, RandSymPyXtal],
                      seeds=Seeds)

from .GenerationController import GenerationController
GenerationController.registerOptimizer(GlobalOptimizer)

from .Calculators.SHELL_Calculator import SHELL_Calculator
from .Calculators.Interfaces.ABINIT_Interface import ABINIT_Interface
ABINIT_Interface.registerTypes(AtomicStructure, Element, Cell, AtomicDisassembler)
SHELL_Calculator.registerInterface('abinit', ABINIT_Interface)
from .Calculators.Interfaces.GULP_Interface import GULP_Interface
GULP_Interface.registerTypes(AtomicStructure, Element, Cell, AtomicDisassembler)
SHELL_Calculator.registerInterface('gulp', GULP_Interface)
from .Calculators.Interfaces.LAMMPS_Interface import LAMMPS_Interface
LAMMPS_Interface.registerTypes(AtomicStructure, Element, Cell, AtomicDisassembler)
SHELL_Calculator.registerInterface('lammps', LAMMPS_Interface)
from .Calculators.Interfaces.MLIP_Interface import MLIP_Interface
MLIP_Interface.registerTypes(AtomicStructure, Element, Cell, AtomicDisassembler)
SHELL_Calculator.registerInterface('mlip', MLIP_Interface)
from .Calculators.Interfaces.PWmat_Interface import PWmat_Interface
PWmat_Interface.registerTypes(AtomicStructure, Element, Cell, AtomicDisassembler)
from .Calculators.Interfaces.QE_Interface import QE_Interface
QE_Interface.registerTypes(AtomicStructure, Element, Cell, AtomicDisassembler)
SHELL_Calculator.registerInterface('qe', QE_Interface)
from .Calculators.Interfaces.VASP_Interface import VASP_Interface
VASP_Interface.registerTypes(AtomicStructure, Element, Cell, AtomicDisassembler)
SHELL_Calculator.registerInterface('vasp', VASP_Interface)
from .Calculators.Interfaces.MOPAC_Interface import MOPAC_Interface
MOPAC_Interface.registerTypes(AtomicStructure, Element, Cell, AtomicDisassembler)
SHELL_Calculator.registerInterface('mopac', MOPAC_Interface)
from .Calculators.Interfaces.FHIaims_Interface import FHIaims_Interface
FHIaims_Interface.registerTypes(AtomicStructure, Element, Cell, AtomicDisassembler)
SHELL_Calculator.registerInterface('aims', FHIaims_Interface)
from .Calculators.TaskManagers.BSUB import BSUB
SHELL_Calculator.registerTaskManager('BSUB', BSUB)
from .Calculators.TaskManagers.QSUB import QSUB
SHELL_Calculator.registerTaskManager('QSUB', QSUB)
from .Calculators.TaskManagers.SBATCH import SBATCH
SHELL_Calculator.registerTaskManager('SBATCH', SBATCH)
from .Calculators.TaskManagers.SHELL import SHELL
SHELL_Calculator.registerTaskManager('SHELL', SHELL)