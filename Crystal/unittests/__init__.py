from .. import Crystal
import os

PREFIX = os.path.dirname(os.path.abspath(__file__))

with open(PREFIX + '/Mg4Al8O16_system1', 'rt') as f:
    Mg4Al8O16_system1 = Crystal.fromJSON(f.read())
    Mg4Al8O16_system1.ID = 0

with open(PREFIX + '/Mg4Al8O16_system2', 'rt') as f:
    Mg4Al8O16_system2 = Crystal.fromJSON(f.read())
    Mg4Al8O16_system2.ID = 0

with open(PREFIX + '/Mg4Al8O16_system3', 'rt') as f:
    Mg4Al8O16_system3 = Crystal.fromJSON(f.read())
    Mg4Al8O16_system3.ID = 0

with open(PREFIX + '/Si2O5_system1', 'rt') as f:
    Si2O5_system1 = Crystal.fromJSON(f.read())
    Si2O5_system1.ID = 0

with open(PREFIX + '/YBaCo4O7_system1', 'rt') as f:
    YBaCo4O7_system1 = Crystal.fromJSON(f.read())
    YBaCo4O7_system1.ID = 0

with open(PREFIX + '/YBaCo4O7_system2', 'rt') as f:
    YBaCo4O7_system2 = Crystal.fromJSON(f.read())
    YBaCo4O7_system2.ID = 0

with open(PREFIX + '/YBaCo4O7_system3', 'rt') as f:
    YBaCo4O7_system3 = Crystal.fromJSON(f.read())
    YBaCo4O7_system3.ID = 0

with open(PREFIX + '/glycine1', 'rt') as f:
    glycine1 = Crystal.fromJSON(f.read())
    glycine1.ID = 0

with open(PREFIX + '/glycine2', 'rt') as f:
    glycine2 = Crystal.fromJSON(f.read())
    glycine2.ID = 0

with open(PREFIX + '/O2_mag1', 'rt') as f:
    O2_mag1 = Crystal.fromJSON(f.read())
    O2_mag1.ID = 0

with open(PREFIX + '/O2_mag2', 'rt') as f:
    O2_mag2 = Crystal.fromJSON(f.read())
    O2_mag2.ID = 0