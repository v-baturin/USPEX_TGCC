import logging
import yaml
import sys
from copy import copy

from .RawParser import parse, ParseError


logger = logging.getLogger(__name__)


def read(filename):
    with open(filename, 'rt') as f:
        definitions = yaml.safe_load(f.read())
    main = definitions.pop('main')
    for name, definition in definitions.items():
        definition['name'] = name
    return _process(main, definitions)


def _process(input, definitions: dict):
    if isinstance(input, str):
        if input in definitions:
            input = copy(definitions[input])
        elif len(input) > 1 and input[0] == '(' and input[-1] == ')':
            try:
                input = list(parse(input))
            except ParseError as e:
                logger.error(f'Error while parsing {input}')
                exc_info = sys.exc_info()
                raise exc_info[0].with_traceback(exc_info[1], exc_info[2])

    if isinstance(input, list):
        items = enumerate(input)
    elif isinstance(input, dict):
        items = input.items()
    else:
        items = []
    for i, element in items:
        if i is not 'name':
            input[i] = _process(element, definitions)
    return input


def write(filename, params):
    with open(filename, 'wt') as f:
        f.write(yaml.safe_dump(params))
