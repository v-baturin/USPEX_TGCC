from .Atomistic.Element import Element
from .Atomistic.CellUtility import Cell
from .Atomistic.AtomicPrimitives import AtomicStructure, AtomicDisassembler
from .Atomistic.AtomisticPoolEntry import AtomisticPoolEntry
AtomisticPoolEntry.registerTypes(AtomicStructure, Element, Cell, AtomicDisassembler)
from .IO.AtomisticRepresentation import AtomisticRepresentation
AtomisticRepresentation.registerTypes(AtomicStructure, Element, Cell, AtomicDisassembler)
from .Optimizers.GlobalOptimizer import GlobalOptimizer
from .Fitness.Fitness import Fitness
GlobalOptimizer.setFitnessType(Fitness)
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
# from .Atomistic.Environments.NanoparticleCore import NanoparticleCore
# NanoparticleCore.registerTypes(AtomisticRepresentation, AtomicStructure, Element, Cell, AtomicDisassembler)
from .Atomistic.RadialDistributionUtility import RadialDistributionUtility
from .Atomistic.CellUtility import CellUtility
from .Atomistic.SimpleMoleculeUtility import SimpleMoleculeUtility
SimpleMoleculeUtility.registerTypes(AtomicStructure, Element)
from .Atomistic.JunctionUtility import JunctionUtility
from .Atomistic.Conditions import Conditions
from .Atomistic.BondUtility import BondUtility
BondUtility.registerTypes(Element, AtomicDisassembler)
# from .Atomistic.ElasticML import ElasticML
# ElasticML.registerTypes(AtomicDisassembler)
from .Atomistic.Constraints import Constraints
# from .XRay.PowderSpectrumAnalyzer import PowderSpectrumAnalyzer
# from .XRay.SingleCrystalSpectrumAnalyzer import SingleCrystalSpectrumAnalyzer
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
# from .Atomistic.Operators.CoreAdsorbantRandomGenerator import CoreAdsorbantRandomGenerator
Seeds.registerTypes(AtomisticRepresentation)
GlobalOptimizer.registerTarget('Atomistic',
                      utilities=[CompositionSpace, RadialDistributionUtility, CellUtility, EnvironmentUtility,
                                 SimpleMoleculeUtility, Conditions, BondUtility, Constraints, JunctionUtility],
                                 # ElasticML,
                                 # PowderSpectrumAnalyzer, SingleCrystalSpectrumAnalyzer],
                      hybridizations=[Heredity],
                      mutations=[Softmodemutation, Permutation, Transmutation, AddAtom, RemoveAtom, TeleportAtom],
                      # creations=[RandTop, RandSym, RandSymPyXtal, CoreAdsorbantRandomGenerator],
                      entry=AtomisticPoolEntry,
                      creations=[RandTop, RandSym, RandSymPyXtal], # CoreAdsorbantRandomGenerator],
                      seeds=Seeds)
from .Stages.Executor import Executor
from .Stages.Interfaces.ASEInterfaceAdapter import ASEInterfaceAdapter
ASEInterfaceAdapter.registerTypes(AtomicStructure, Element, Cell)
from .Stages.Interfaces.ABINIT_Interface import ABINIT_Interface
ABINIT_Interface.registerTypes(AtomicStructure, Element, Cell)
Executor.registerInterface('abinit', ABINIT_Interface)
from .Stages.Interfaces.GULP_Interface import GULP_Interface
GULP_Interface.registerTypes(AtomicStructure, Element, Cell)
Executor.registerInterface('gulp', GULP_Interface)
# from .Stages.Interfaces.LAMMPS_Interface import LAMMPS_Interface
# LAMMPS_Interface.registerTypes(AtomisticRepresentation, ASEInterfaceAdapter.LAMMPS)
# Executor.registerInterface('lammps', LAMMPS_Interface)
# from .Stages.Interfaces.MLIP_Interface import MLIP_Interface
# MLIP_Interface.registerTypes(AtomisticRepresentation, AtomicDisassembler)
# Executor.registerInterface('mlip', MLIP_Interface)
# from .Stages.Interfaces.PWmat_Interface import PWmat_Interface
# PWmat_Interface.registerTypes(AtomicStructure, Element, Cell)
# from .Stages.Interfaces.QE_Interface import QE_Interface
# QE_Interface.registerTypes(ASEInterfaceAdapter.QE)
# Executor.registerInterface('qe', QE_Interface)
from .Stages.Interfaces.VASP_Interface import VASP_Interface
VASP_Interface.registerTypes(ASEInterfaceAdapter.VASP)
Executor.registerInterface('vasp', VASP_Interface)
from .Stages.Interfaces.MOPAC_Interface import MOPAC_Interface
MOPAC_Interface.registerTypes(AtomicStructure, Element, Cell)
Executor.registerInterface('mopac', MOPAC_Interface)
# from .Stages.Interfaces.FHIaims_Interface import FHIaims_Interface
# FHIaims_Interface.registerTypes(AtomicStructure, Element, Cell)
# Executor.registerInterface('aims', FHIaims_Interface)
# from .Stages.Interfaces.CP2K_Interface import CP2K_Interface
# CP2K_Interface.registerTypes(AtomicStructure, Element, Cell)
# Executor.registerInterface('cp2k', CP2K_Interface)
from .Stages.TaskManagers.BSUB import BSUB
Executor.registerTaskManager('BSUB', BSUB)
from .Stages.TaskManagers.QSUB import QSUB
Executor.registerTaskManager('QSUB', QSUB)
from .Stages.TaskManagers.SBATCH import SBATCH
Executor.registerTaskManager('SBATCH', SBATCH)
from .Stages.TaskManagers.TGCC import TGCC
Executor.registerTaskManager('TGCC', TGCC)
from .Stages.TaskManagers.SHELL import SHELL
Executor.registerTaskManager('SHELL', SHELL)
from .Optimizers.ModelOptimizer import ModelOptimizer, External
External.setExecutorType(Executor)
ModelOptimizer.registerModel(External)
ModelOptimizer.registerTarget('Atomistic',
                      utilities=[CompositionSpace, RadialDistributionUtility, CellUtility, EnvironmentUtility,
                                 SimpleMoleculeUtility, Conditions, BondUtility, Constraints, JunctionUtility],
                                 # ElasticML,
                                 # PowderSpectrumAnalyzer, SingleCrystalSpectrumAnalyzer],
                      hybridizations=[Heredity],
                      mutations=[Softmodemutation, Permutation, Transmutation, AddAtom, RemoveAtom, TeleportAtom],
                      creations=[RandTop, RandSym, RandSymPyXtal], # CoreAdsorbantRandomGenerator],
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