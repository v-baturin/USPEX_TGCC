"""
USPEX.Common.System
===================

Class describing system to optimize

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

from copy import copy
import json


class System(object):
    """
    Class describing system to optimize.
    """

    _isBad = False

    def markBad(self):
        """
        Mark system as bad. Systems marked bad are processed differently.
        """
        self.isBad = True

    @property
    def isBad(self):
        """
        Shows if system was marked as bad and should be processed differently from systems not marked as such.

        :rtype: bool
        :return: True if system marked as bad, False otherwise.
        """
        return self._isBad

    @isBad.setter
    def isBad(self, value: bool):
        """
        Setter for property isBad.

        :type value: bool
        :param value: True if we to mark the system as bad, False otherwise.
        """
        self._isBad = value

    def toJSON(self) -> str:
        """
        Create JSON representation of the structure.

        :rtype: str
        :return: String with JSON representation of the structure.
        """
        return json.dumps(self.toDICT())

    def toDICT(self) -> dict:
        """
        Method which creates dictionary representation of the structure.

        :return: Dictionary representing the structure.
        """
        dct = copy(self.__dict__)
        if '__hash__' in dct:
            del dct['__hash__']
        if '__copy__' in dct:
            del dct['__copy__']
        return dct

    @classmethod
    def fromJSON(cls, repr: str):
        """
        Reconstruct :class:`AtomicStructure` from JSON representation.

        :type repr: str
        :param repr: String with JSON representation of the structure.
        :rtype: type corresponding to this classmethod. (see Examples)
        :return: new created structure.
        """
        return cls.fromDICT(json.loads(repr))

    @classmethod
    def fromDICT(cls, dct: dict):
        """
        Method which reconstructs AtoimicStructure from dictionary representation.

        :type dct: dict
        :param dct: dictionary representing the structure.
        """
        newStructure = cls()
        newStructure.__dict__.update(copy(dct))
        return newStructure

    def copy(self):
        """
        Method which returns a copy of the system.
        """
        dct = self.toDICT()
        del dct['ID']
        return type(self).fromDICT(dct)

    def _systemHash(self):
        return hash(self.ID)
