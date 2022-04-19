from copy import copy
from pprint import pformat

from .RawParser import parse


def read(filename):
    with open(filename, 'rt') as f:
        sections = f.read().split('#define ')
    definitions = {}
    for section in sections[1:]:
        name, definition = section.split('\n', 1)
        definitions[name] = parse(definition)
    return _process(parse(sections[0]), definitions)


def _process(input, definitions: dict):
    if isinstance(input, str) and input in definitions:
        input = copy(definitions[input])
    if isinstance(input, list):
        items = enumerate(input)
    elif isinstance(input, dict):
        items = input.items()
    else:
        items = []
    for i, element in items:
        input[i] = _process(element, definitions)
    return input


def write(filename, params):
    content = f"{pformat(params, width=120)}\n"
    with open(filename, 'wt') as f:
        f.write(content)
