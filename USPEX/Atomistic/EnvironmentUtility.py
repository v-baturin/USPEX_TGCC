"""
USPEX.Atomistic.EnvironmentUtility
==================================
"""

from .Environments.Interface import Interface
from .Environments.Substrate import Substrate
from .Environments.Bulk import Bulk
from .Environments.NanoparticleCore import NanoparticleCore

class EnvironmentUtility:
    """
    Class representing utility which generates possible environments for calculation.
    """

    supportedEnvironments = {
        'interface': Interface,
        'substrate': Substrate,
        'bulk': Bulk,
        'nanoparticle_core': NanoparticleCore
    }


    @classmethod
    def build(cls, type, **description):
        return cls.supportedEnvironments.get(type).Assembler.build(**description)

    def __init__(self, environments: list = None):
        """

        """
        self.assemblers = [self.supportedEnvironments.get(environment['type']).Assembler(**environment)
                           for environment in (environments if environments is not None else [])]


