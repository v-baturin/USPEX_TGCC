"""
USPEX.Common.VarOperator
========================

Basic class for variation operators

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
.. codeauthor:: Artem Samtsevich <samtsevichartem@gmail.com>
"""

from typing import Callable

from USPEX.Common.Config import Config
from USPEX.Common.SystemPool import SystemPool

MAX_OUTPUT_SIZE = 8


class VarOperator(object):
    """
    Variation Operators are wise methods of creation of new systems in some target space (see
    :class:`~USPEX.Common.Target.Target`). Such operators can be of three types: hybridization, mutation and creation.

    * Hybridization takes two parent systems and somehow combines their features giving birth to new system.
    * Mutation takes one system and changes its features creating new one.
    * Creation just makes a system from nowhere. Actually from our understanding of the nature of target space.

    Operator of any of these types actually can be fed with arbitrary set of structures (see :meth:`tune`)
    to learn something about the whole target space or rather its known part.

    :class:`VarOperator` instances are callable (see :meth:`__call__`).

    :examples:

    If **Heredity** is some descendant of :class:`VarOperator`.

    >>> heredity = Heredity(**params)
    >>> offsprings = heredity(system1, system2)

    """

    name = None
    isActive = True

    def __init__(self, systemFactory: Callable, config: dict, pool: SystemPool, initFrac: float, minFrac: float = 0.1, maxFrac: float = 1.0,
                 maxOutputSize: int = MAX_OUTPUT_SIZE):
        """
        Initializes the class.

        :type systemFactory: Callable
        :param systemFactory: factory to be used for new system creation.
        :type config: dict
        :param config: dictionary of parameters providied by user for new system creation.
        :type pool: :class:`~USPEX.Common.SystemPool.SystemPool` or descendant
        :param pool: system pool.
        :type initFrac: float
        :param initFrac: initial fraction of this variation operator in search algorithm.
        :type minFrac: float
        :param minFrac: minimal fraction of this variation operator in search algorithm.
        :type maxFrac: float
        :param maxFrac: maximal fraction of this variation operator in search algorithm.
        :type maxOutputSize: int
        :param maxOutputSize: maximum number of offsprings this operator can give from single execution.
        """

        self.systemFactory = systemFactory
        self.config = config
        self.pool = pool
        assert 0.0 <= minFrac <= 1.0
        assert 0.0 <= maxFrac <= 1.0
        assert minFrac <= maxFrac
        self._minimalFraction = minFrac
        self._maximalFraction = maxFrac
        self._initialFraction = initFrac
        self._MAX_OUTPUT_SIZE = int(maxOutputSize)

    def __str__(self):
        return self.name

    @property
    def initialFraction(self):
        """
        Returns the initial fraction of this variation operator in search algorithm.
        """
        return self._initialFraction

    @property
    def minimalFraction(self):
        """
        Returns the minimal fraction of this variation operator in search algorithm.
        """
        return self._minimalFraction

    @property
    def maximalFraction(self):
        """
        Returns the maximal fraction of this variation operator in search algorithm.
        """
        return self._maximalFraction

    @property
    def maxOutputSize(self) -> int:
        """
        Returns the maximum number of offsprings this operator can give from single execution.
        """
        return self._MAX_OUTPUT_SIZE

    def __hash__(self):
        return hash(self.name)

    def __call__(self, *args, **kwargs) -> tuple:
        """
        Body of variation operator.

        :type args: list
        :param args:
            parents for new system. Two arguments for hybridization, one for mutation, none for creation.
        :type kwargs: dict
        :param kwargs:
            optional parameters. These are used only for testing to fix some parameters,
            when you do not want stochasticity.
        :rtype: tuple
        :return: offsprings.
        :raises VOFailed: Should be raised if variation fails.
        """
        return ()

    def tune(self, population : list):
        """
        Tune variation operator by providing data about some set of known systems.

        :type population: list of :class:`~USPEX.Common.System.System` descendants.
        :param population: list of systems which can be used to tune variation operator behaviour.
        """
        pass

    def prepare(self):
        pass

    def standby(self):
        pass


class VOFailed(Exception):
    """
    Should be raised when variation operator fails.
    """
    pass
