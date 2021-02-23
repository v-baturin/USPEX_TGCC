from .Target import Target
from .Atomistic.Crystal import Crystal
from .Atomistic.CompositionSpace import CompositionSpace
from .Atomistic.RadialDistributionUtility import RadialDistributionUtility
from .XRay.SpectrumAnalyzer import SpectrumAnalyzer
from .Crystal import variationOperators
Target.registerTarget('Crystal', Crystal, [CompositionSpace, SpectrumAnalyzer, RadialDistributionUtility], variationOperators)

from .GlobalOptimizer import GlobalOptimizer
from .Fitness import Fitness
from .Selection.USPEXClassic import USPEXClassic
GlobalOptimizer.setFitnessType(Fitness)
GlobalOptimizer.registerSelection(USPEXClassic)

from .Controllers.GenerationController import GenerationController
GenerationController.registerOptimizer(GlobalOptimizer)

from .InputParser import read
from .IO.compileParams import compileParams
