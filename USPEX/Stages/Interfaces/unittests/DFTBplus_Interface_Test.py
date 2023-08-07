import unittest
import numpy as np
from pathlib import Path

from ....Optimizers.PoolEntry import PoolEntry, EntryFlavour
from ....components import AtomisticRepresentation, DFTBplus_Interface, Atomistic

HOMEPATH = Path(__file__).parent
SPECIFICPATH = HOMEPATH/'dftbSpecific'
GATHEREDPATH = HOMEPATH/'dftbGatheredData'

class DFTBplus_InterfaceTest(unittest.TestCase):
    def test_read_output(self):
        ID = 0
        # HERE what is written in dftb_in.hsd does not make sense.
        # Only output will be parsed and properties checked
        interface = DFTBplus_Interface(tag='0', dftb_input=SPECIFICPATH/'dftb_in.hsd_1', kresol=0.04)
        atomistic = Atomistic()
        extensions = dict(
            atomistic=atomistic.propertyExtension(atomistic)
        )
        structure = AtomisticRepresentation.readPOSCAR(GATHEREDPATH/f'input/system{ID}.vasp', (1, 1, 1))
        disassembler = AtomisticRepresentation.atomicDisassemblerType(np.arange(len(structure)).reshape((-1, 1)))
        system = PoolEntry(ID, EntryFlavour(extensions=extensions))
        system.setProperty('externalPressure', 0.0)
        system.setProperty('disassembler', disassembler, extension='atomistic', suffix='intermediate')
        system.setProperty('disassembler', disassembler, extension='atomistic', suffix='0')
        system.setProperty('structure', structure, extension='atomistic', suffix='intermediate')
        interface.readOutput(system=system, calcFolder=GATHEREDPATH/f'output/CalcFold{ID}')
        self.assertTrue(np.isclose(system['.enthalpy.0'], -395.547))