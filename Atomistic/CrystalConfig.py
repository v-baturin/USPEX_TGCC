"""
USPEX.Common.Atomistic.CrystalConfig
====================================

Class for description of configuration of crystal target space.

.. codeauthor:: Artem Samtsevich <samtsevichartem@gmail.com>
.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

from .AtomisticConfig import AtomisticConfig
from .Crystal import Crystal
from ..XRay.SpectrumAnalyzer import SpectrumAnalyzer


class CrystalConfig(AtomisticConfig):
    """
    Class describing configuration space for Crystal structures.
    Descendant of :class:`~USPEX.Common.Atomistic.AtomisticConfig.AtomisticConfig`.
    """

    def __init__(self, isConstLattice: bool=False, latticeValues=None, xraydata=None, **kwargs):
        """
        Initializes the class.

        :type isConstLattice: bool
        :param isConstLattice: True if constant lattice, False otherwise.
        :type latticeValues: list
        :param latticeValues: specifies known lattice parameters.
        :type kwargs: dict
        :param kwargs: additional arguments and keywords used to initialize the parent class AtomisticConfig.
        """
        #TODO
        # * latticeValues

        super(CrystalConfig, self).__init__(**kwargs)
        self.isConstLattice = isConstLattice
        self.latticeValues = latticeValues
        self.xraydata = xraydata

        if xraydata is not None:
            assert isinstance(xraydata, dict)
            self._spectrumAnalyzer = SpectrumAnalyzer(**xraydata)

    def isGoodSystem(self, system) -> bool:
        """
        Method which checks if the structure meets composition constraints.

        :type system: :class:`~USPEX.Common.Atomistic.AtomicStructure.AtomicStructure` descendant
        :param system: structure to check.
        :rtype: bool
        :return: True if the structure meets composition constraints, False otherwise.
        """
        return super().isGoodSystem(system) and self.isGoodLattice(system)

    def xraydistance(self, system):
        """
        Method which takes a structure and calculates the distance (fitness function)
        between calculated and experimental X-ray spectrum.

        :type system: :class:`~USPEX.Common.Atomistic.AtomicStructure.AtomicStructure` descendant
        :param system: structure to compare with experiment.
        :rtype: float
        :return: distance between calculated and experimental spectrum.
        """
        if hasattr(self, '_spectrumAnalyzer'):
            return self._spectrumAnalyzer[system]
        else:
            raise RuntimeError('Cannot optimize the quantity xraydistance. No experimental X-ray data found.')

    @property
    def systemFactory(self):
        """
        A reference to the class representing the system in this configuration space.

        :rtype: :class:`Crystal`
        :return: system class.
        """
        return Crystal
