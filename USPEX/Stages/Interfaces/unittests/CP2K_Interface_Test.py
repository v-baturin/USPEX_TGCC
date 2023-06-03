import unittest
import numpy as np
from pathlib import Path

from ....components import AtomisticRepresentation, CP2K_Interface, AtomisticPoolEntry

HOMEPATH = Path(__file__).parent
SPECIFICPATH = HOMEPATH/'cp2kSpecific'
GATHEREDPATH = HOMEPATH/'cp2kGatheredData'

class CP2K_InterfaceTest(unittest.TestCase):
    def test_read_output(self):
        ID = 0
        # HERE what is written in cp2k.inp does not make sense.
        # Only output will be parsed and properties checked
        interface = CP2K_Interface(tag='0', cp2k_in=SPECIFICPATH/'cp2k.inp_1', kresol=0.12)
        structure = AtomisticRepresentation.readPOSCAR(GATHEREDPATH/f'input/system{ID}.vasp', (1, 1, 1))
        system = AtomisticPoolEntry(
            ID=ID,
            structure=structure,
            disassembler=AtomisticRepresentation.atomicDisassemblerType(
                np.arange(len(structure)).reshape((-1, 1))),
            externalPressure=0.0
        )
        interface.readOutput(system=system, calcFolder=GATHEREDPATH/f'output/CalcFold{ID}')
        self.assertTrue(np.isclose(system['enthalpy'], -404.816))