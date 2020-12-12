from copy import copy
from ast import literal_eval
from pprint import pformat


def read(filename):
    with open(filename, 'rt') as f:
        content = f.read()
    sections = content.split('#define ')

    try:
        definitions = {
            'main': literal_eval(sections.pop(0))
        }
    except SyntaxError:
        # TODO add logger
        definitions = {}
    for section in sections:
        name, definition = section.split('\n', 1)
        assert name != 'main'
        definitions[name] = literal_eval(definition)

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