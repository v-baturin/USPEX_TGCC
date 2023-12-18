from .Optimizers.PoolEntry import PoolEntry
PoolEntry.createEngine("uspex.db")
# ---------------------------------- Primitives and Representations -------------------------------------------------
from .Atomistic.Primitives.Element import Element
from .Atomistic.Primitives.Cell import Cell
from .Atomistic.Primitives.AtomicStructure import AtomicStructure
from .IO.AtomicStructureRepresentation import AtomicStructureRepresentation
from .Atomistic.Atomistic import Atomistic
Atomistic.registerTypes(AtomicStructure, Element, Cell, AtomicStructureRepresentation)
# --------------------------------------------- Atomistic -----------------------------------------------------------
from .Atomistic.CompositionSpace import CompositionSpace
from .Atomistic.EnvironmentUtility import EnvironmentUtility
from .Atomistic.Environments.Interface import Interface
Interface.registerTypes(Atomistic)
from .Atomistic.Environments.Substrate import Substrate
Substrate.registerTypes(Atomistic)
from .Atomistic.Environments.Bulk import Bulk
Bulk.registerTypes(Atomistic)
from .Atomistic.Environments.NanoparticleCore import NanoparticleCore
NanoparticleCore.registerTypes(Atomistic)
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
# --------------------------------------------- Targets--- ----------------------------------------------------------
from .Optimizers.Target import Target
Target.registerTarget('Atomistic',
                      utilities=[Atomistic, CompositionSpace, RadialDistributionUtility, CellUtility,
                        EnvironmentUtility, SimpleMoleculeUtility, Conditions, BondUtility, ElasticML,
                        PowderSpectrumAnalyzer, SingleCrystalSpectrumAnalyzer, JunctionUtility],
                      hybridizations=[Heredity],
                      mutations=[Softmodemutation, Permutation, Transmutation, AddAtom, RemoveAtom, TeleportAtom],
                      creations=[RandTop, RandSym, RandSymPyXtal, CoreAdsorbantRandomGenerator],
                      seeds=Seeds,
                      defaultMetric='radialDistributionUtility')
# --------------------------------------------- Interfaces ----------------------------------------------------------
from .Stages.Executor import Executor
from .Stages.Interfaces.ABINIT_Interface import ABINIT_Interface
Executor.registerInterface('abinit', ABINIT_Interface)
from .Stages.Interfaces.GULP_Interface import GULP_Interface
Executor.registerInterface('gulp', GULP_Interface)
from .Stages.Interfaces.LAMMPS_Interface import LAMMPS_Interface
LAMMPS_Interface.registerTypes(AtomicStructureRepresentation)
Executor.registerInterface('lammps', LAMMPS_Interface)
from .Stages.Interfaces.MLIP_Interface import MLIP_Interface
Executor.registerInterface('mlip', MLIP_Interface)
from .Stages.Interfaces.QE_Interface import QE_Interface
QE_Interface.registerTypes(AtomicStructureRepresentation)
Executor.registerInterface('qe', QE_Interface)
from .Stages.Interfaces.VASP_Interface import VASP_Interface
VASP_Interface.registerTypes(AtomicStructureRepresentation)
Executor.registerInterface('vasp', VASP_Interface)
from .Stages.Interfaces.MOPAC_Interface import MOPAC_Interface
Executor.registerInterface('mopac', MOPAC_Interface)
from .Stages.Interfaces.FHIaims_Interface import FHIaims_Interface
Executor.registerInterface('aims', FHIaims_Interface)
from .Stages.Interfaces.XTB_Interface import XTB_Interface
XTB_Interface.registerTypes(AtomicStructureRepresentation)
Executor.registerInterface('xtb', XTB_Interface)
from .Stages.Interfaces.DFTBplus_Interface import DFTBplus_Interface
DFTBplus_Interface.registerTypes(AtomicStructureRepresentation)
Executor.registerInterface('dftb', DFTBplus_Interface)
from .Stages.Interfaces.CP2K_Interface import CP2K_Interface
Executor.registerInterface('cp2k', CP2K_Interface)
# -------------------------------------------- Task Managers --------------------------------------------------------
from .Stages.TaskManagers.BSUB import BSUB
Executor.registerTaskManager('BSUB', BSUB)
from .Stages.TaskManagers.QSUB import QSUB
Executor.registerTaskManager('QSUB', QSUB)
from .Stages.TaskManagers.SBATCH import SBATCH
Executor.registerTaskManager('SBATCH', SBATCH)
from .Stages.TaskManagers.SHELL import SHELL
Executor.registerTaskManager('SHELL', SHELL)
from .Stages.AtomisticStage import AtomisticStage
# ------------------------------------------------ Stages -----------------------------------------------------------
AtomisticStage.registerTypes(Executor)
from .Stages.PopulationProcessor import PopulationProcessor, Stages
Stages.registerStage('execute', Executor)
Stages.registerStage('atomistic', AtomisticStage)
Stages.registerStage('populationProcessor', PopulationProcessor)
# ---------------------------------------- Generation Controller ----------------------------------------------------
from .Stages.GenerationController import GenerationController
from .Optimizers.GlobalOptimizer import GlobalOptimizer
GenerationController.registerOptimizer(GlobalOptimizer)
from .Generators.Evolution import Evolution
Evolution.setTarget(Target)
GenerationController.registerGenerator(Evolution)
GenerationController.setPopulationProcessor(PopulationProcessor)
from .IO.compileParams import compileParams
GenerationController.setUpcompileParams(compileParams)
# ---------------------------------------- Output Representation ----------------------------------------------------
from .IO.AtomisticRepresentation import AtomisticRepresentation
AtomisticRepresentation.registerTypes(Atomistic)
