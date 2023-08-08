from .Atomistic.Primitives.Element import Element
from .Atomistic.Primitives.Cell import Cell
from .Atomistic.Primitives.AtomicStructure import AtomicStructure
from .Atomistic.Atomistic import Atomistic, AtomicDisassembler
Atomistic.registerTypes(AtomicStructure, Element, Cell)
from .IO.AtomisticRepresentation import AtomisticRepresentation
AtomisticRepresentation.registerTypes(AtomicStructure, Element, Cell, AtomicDisassembler)
from .Optimizers.GlobalOptimizer import GlobalOptimizer
from .Expressions.ExpressionEvaluator import ExpressionEvaluator
GlobalOptimizer.setExpressionEvaluatorType(ExpressionEvaluator)
from .Selection.USPEXClassic import USPEXClassic
GlobalOptimizer.registerSelection(USPEXClassic)
from .Atomistic.CompositionSpace import CompositionSpace
from .Atomistic.EnvironmentUtility import EnvironmentUtility
from .Atomistic.Environments.Interface import Interface
Interface.registerTypes(AtomisticRepresentation, AtomicStructure, Element, Cell, AtomicDisassembler)
from .Atomistic.Environments.Substrate import Substrate
Substrate.registerTypes(AtomisticRepresentation, AtomicStructure, Element, Cell, AtomicDisassembler)
from .Atomistic.Environments.Bulk import Bulk
Bulk.registerTypes(AtomisticRepresentation, AtomicStructure, Element, Cell, AtomicDisassembler)
from .Atomistic.Environments.NanoparticleCore import NanoparticleCore
NanoparticleCore.registerTypes(AtomisticRepresentation, AtomicStructure, Element, Cell, AtomicDisassembler)
from .Atomistic.RadialDistributionUtility import RadialDistributionUtility
from .Atomistic.CellUtility import CellUtility
from .Atomistic.SimpleMoleculeUtility import SimpleMoleculeUtility
SimpleMoleculeUtility.registerTypes(AtomicStructure, Element)
from .Atomistic.JunctionUtility import JunctionUtility
from .Atomistic.Conditions import Conditions
from .Atomistic.BondUtility import BondUtility
from .Atomistic.ElasticML import ElasticML
from .XRay.PowderSpectrumAnalyzer import PowderSpectrumAnalyzer
from .XRay.SingleCrystalSpectrumAnalyzer import SingleCrystalSpectrumAnalyzer
from .Atomistic.Operators.Heredity import Heredity
from .Atomistic.Operators.RandTop import RandTop
from .Atomistic.Operators.RandSym import RandSym
from .Atomistic.Operators.RandSymPyXtal import RandSymPyXtal
from .Atomistic.Operators.Softmodemutation import Softmodemutation
from .Atomistic.Operators.Permutation import Permutation
from .Atomistic.Operators.Transmutation import Transmutation
from .Atomistic.Operators.AddAtom import AddAtom
from .Atomistic.Operators.RemoveAtom import RemoveAtom
from .Atomistic.Operators.TeleportAtom import TeleportAtom
from .Atomistic.Operators.Seeds import Seeds
from .Atomistic.Operators.CoreAdsorbantRandomGenerator import CoreAdsorbantRandomGenerator
Seeds.registerTypes(AtomisticRepresentation)
GlobalOptimizer.registerTarget('Atomistic',
                               utilities=[Atomistic, CompositionSpace, RadialDistributionUtility, CellUtility,
                                 EnvironmentUtility, SimpleMoleculeUtility, Conditions, BondUtility, ElasticML,
                                 PowderSpectrumAnalyzer, SingleCrystalSpectrumAnalyzer, JunctionUtility],
                               hybridizations=[Heredity],
                               mutations=[Softmodemutation, Permutation, Transmutation, AddAtom, RemoveAtom, TeleportAtom],
                               creations=[RandTop, RandSym, RandSymPyXtal, CoreAdsorbantRandomGenerator],
                               seeds=Seeds)
from .Stages.Executor import Executor
from .Stages.Interfaces.ASEInterfaceAdapter import ASEInterfaceAdapter
ASEInterfaceAdapter.registerTypes(AtomicStructure, Element, Cell)
from .Stages.Interfaces.ABINIT_Interface import ABINIT_Interface
Executor.registerInterface('abinit', ABINIT_Interface)
from .Stages.Interfaces.GULP_Interface import GULP_Interface
Executor.registerInterface('gulp', GULP_Interface)
from .Stages.Interfaces.LAMMPS_Interface import LAMMPS_Interface
LAMMPS_Interface.registerTypes(AtomisticRepresentation, ASEInterfaceAdapter.LAMMPS)
Executor.registerInterface('lammps', LAMMPS_Interface)
from .Stages.Interfaces.MLIP_Interface import MLIP_Interface
MLIP_Interface.registerTypes(AtomisticRepresentation)
Executor.registerInterface('mlip', MLIP_Interface)
from .Stages.Interfaces.QE_Interface import QE_Interface
QE_Interface.registerTypes(ASEInterfaceAdapter.QE)
Executor.registerInterface('qe', QE_Interface)
from .Stages.Interfaces.VASP_Interface import VASP_Interface
VASP_Interface.registerTypes(ASEInterfaceAdapter.VASP)
Executor.registerInterface('vasp', VASP_Interface)
from .Stages.Interfaces.MOPAC_Interface import MOPAC_Interface
Executor.registerInterface('mopac', MOPAC_Interface)
from .Stages.Interfaces.FHIaims_Interface import FHIaims_Interface
Executor.registerInterface('aims', FHIaims_Interface)
from .Stages.Interfaces.XTB_Interface import XTB_Interface
XTB_Interface.registerTypes(ASEInterfaceAdapter.GEN)
Executor.registerInterface('xtb', XTB_Interface)
from .Stages.Interfaces.DFTBplus_Interface import DFTBplus_Interface
DFTBplus_Interface.registerTypes(ASEInterfaceAdapter.GEN)
Executor.registerInterface('dftb', DFTBplus_Interface)
from .Stages.Interfaces.CP2K_Interface import CP2K_Interface
Executor.registerInterface('cp2k', CP2K_Interface)
from .Stages.TaskManagers.BSUB import BSUB
Executor.registerTaskManager('BSUB', BSUB)
from .Stages.TaskManagers.QSUB import QSUB
Executor.registerTaskManager('QSUB', QSUB)
from .Stages.TaskManagers.SBATCH import SBATCH
Executor.registerTaskManager('SBATCH', SBATCH)
from .Stages.TaskManagers.SHELL import SHELL
Executor.registerTaskManager('SHELL', SHELL)
from .Optimizers.ModelOptimizer import ModelOptimizer, External
External.setExecutorType(Executor)
ModelOptimizer.registerModel(External)
ModelOptimizer.registerTarget('Atomistic',
                      utilities=[Atomistic, CompositionSpace, RadialDistributionUtility, CellUtility,
                                 EnvironmentUtility, SimpleMoleculeUtility, Conditions, BondUtility, ElasticML,
                                 PowderSpectrumAnalyzer, SingleCrystalSpectrumAnalyzer, JunctionUtility],
                      hybridizations=[Heredity],
                      mutations=[Softmodemutation, Permutation, Transmutation, AddAtom, RemoveAtom, TeleportAtom],
                      creations=[RandTop, RandSym, RandSymPyXtal, CoreAdsorbantRandomGenerator],
                      seeds=Seeds)
from .Stages.AtomisticStage import AtomisticStage
AtomisticStage.registerTypes(Executor)
from .Stages.PopulationProcessor import PopulationProcessor, Stages
from .Stages.GenerationController import GenerationController
GenerationController.registerOptimizer(GlobalOptimizer)
GenerationController.registerOptimizer(ModelOptimizer)
Stages.registerStage('execute', Executor)
Stages.registerStage('atomistic', AtomisticStage)
Stages.registerStage('populationProcessor', PopulationProcessor)
GenerationController.setPopulationProcessor(PopulationProcessor)
from .IO.compileParams import compileParams
GenerationController.setUpcompileParams(compileParams)