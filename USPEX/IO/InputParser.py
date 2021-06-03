from copy import copy
from pprint import pformat

from .RawParser import parse


def read(filename):
    with open(filename, 'rt') as f:
        content = f.read()
    sections = content.split('#define ')

    definitions = {}
    content = sections.pop(0)
    if content:
        definitions['main'] = parse(content)

    for section in sections:
        name, definition = section.split('\n', 1)
        assert name != 'main'
        definitions[name] = parse(definition)

    return definitions

def write(filename, definitions):
    definitions = copy(definitions)

    if 'main' in definitions:
        content = f"{pformat(definitions['main'], width=120)}\n"
        del definitions['main']
    else:
        content = ""

    for name, definition in definitions.items():
        content += f"#define {name}\n{pformat(definition, width=120)}\n"

    with open(filename, 'wt') as f:
        f.write(content)