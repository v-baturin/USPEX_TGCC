import unittest
import numpy as np
from pathlib import Path

from ..ORCA_Interface import ORCA_Interface
from ....Optimizers.PoolEntry import EntryFlavour
from ....Atomistic.Primitives.Element import Element
from ....Atomistic.Primitives.Cell import Cell
from ....Atomistic.Primitives.AtomicStructure import AtomicStructure
from ....IO.AtomicStructureRepresentation import AtomicStructureRepresentation
from ....Atomistic.Atomistic import Atomistic
Atomistic.registerTypes(AtomicStructure, Element, Cell, AtomicStructureRepresentation)
ORCA_Interface.registerTypes(AtomicStructureRepresentation)


HOMEPATH = Path(__file__).parent
SPECIFICPATH = HOMEPATH/'orcaSpecific'
GATHEREDPATH = HOMEPATH/'orcaGatheredData'

class ORCA_InterfaceTest(unittest.TestCase):

    def test_read_output(self):
        ID = 0
        # HERE what is written in orca.in does not make sense.
        # Only output will be parsed and properties checked
        interface = ORCA_Interface(tag='0', orca_input=SPECIFICPATH/'orca.in_1',
                                  targetProperties=['structure', 'enthalpy'])
        atomistic = Atomistic()
        extensions = dict(
            atomistic=atomistic.propertyExtension()
        )

        structure = AtomicStructureRepresentation.readPOSCAR(GATHEREDPATH/f'input/system{ID}.vasp', (0, 0, 0))
        disassembler = Atomistic.atomicDisassemblerType(np.arange(len(structure)).reshape((-1, 1)))
        intermediate = disassembler.disassemble(structure)
        intermediate['.externalPressure'] = 0.0
        intermediate['atomistic.disassembler'] = disassembler
        intermediate = EntryFlavour(extensions=extensions, **intermediate)
        result = interface.readOutput(intermediate, calcFolder=GATHEREDPATH/f'output/CalcFold{ID}')
        self.assertTrue(np.isclose(result['.enthalpy'], -92847.972))
