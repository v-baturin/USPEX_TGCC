from pathlib import Path
import json

PREFIX = Path(__file__).parent

with open(PREFIX/'Si4system', 'rt') as f:
    Si4System = json.load(f)
    Si4System['ID'] = 0
    Si4System['tag'] = 's0'

with open(PREFIX/'Ar4system', 'rt') as f:
    Ar4System = json.load(f)
    Ar4System['ID'] = 0
    Ar4System['tag'] = 's0'

with open(PREFIX/'Si4system', 'rt') as f:
    Si4System = json.load(f)
    Si4System['ID'] = 0
    Si4System['tag'] = 's0'
