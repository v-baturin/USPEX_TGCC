"""
USPEX.Common.Config
===================

Basic Config class

.. codeauthor:: Artem Samtsevich <samtsevichartem@gmail.com>
"""


class Config(object):
    """
    This interface is used for verification if some system belongs a to particular configuration space.
    Classes implementing this interface will usually contain all information about such configuration space
    (i.e. possible compositions) and some methods for calculating secondary quantities (i.e. unit volume).
    In addition, such classes might implement methods for checking some aspects of system being conformant
    with this space (i.e. isGoodComposition).

    .. note:: To implement this interface you don't need to inherit it. You should only implement all its methods.
     More important you should not check anywhere in the code if some class implementing this interface
     is inherited from this interface.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def isGoodSystem(self, system):
        """
        Check if given system belongs to this configuration space.

        :type system: :class:`~USPEX.Common.Atomistic.AtomicStructure.AtomicStructure` or descendant
        :param system: Some system to be checked against it conformance to this space.
        :rtype: bool
        :return: True if the system belongs to this configuration space, False otherwise.
        """
        return True

    @property
    def systemFactory(self):
        """
        A reference to the class representing the system in this configuration space.

        :rtype: None
        :return: system class.
        """
        return None
