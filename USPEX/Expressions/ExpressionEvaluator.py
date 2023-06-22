"""
USPEX.Common.ExpressionEvaluator
================================

Infrastructure component which evaluates complex expression for a pool of individuals.
It utilizes extensions approach which allows developer to define their own primitives (properties and functions)
to be used in expression.

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import logging
import numpy as np
from typing import Mapping, Sequence, Union

from USPEX.Expressions.Functions.presets import applyPresetsRecursive


logger = logging.getLogger(__name__)


class ExpressionEvaluator:

    @staticmethod
    def calculate(expression: Union[str, tuple, int, float], pool: Sequence, extensions: Mapping) -> None:
        expression = applyPresetsRecursive(expression)
        calculator = ExpressionEvaluator(pool, extensions)
        calculator.evaluate(expression)
        calculator.setAllExpressions()

    def __init__(self, pool: Sequence, extensions: Mapping):
        self._pool = pool
        self._extensions = extensions
        self._storedData = {}

    def evaluate(self, expression: Union[str, tuple, int, float]) -> np.ndarray:
        if expression not in self._storedData:
            if len(self._pool) == 0:
                valueArray = np.empty(0)
            elif isinstance(expression, tuple):
                funcName, *funcParams = expression
                assert isinstance(funcName, str), f'Incorrect type {type(funcName)} of function {funcName}.'
                arguments = [self.evaluate(param) for param in funcParams]
                size = min(len(arg) for arg in arguments if hasattr(arg, '__len__'))
                for i, arg in enumerate(arguments):
                    if hasattr(arg, '__len__'):
                        arguments[i] = arg[:size]
                funcName = funcName.split('.')
                if len(funcName) == 1:
                    extension = 'basic'
                    funcName, = funcName
                elif len(funcName) == 2:
                    extension, funcName = funcName
                else:
                    raise RuntimeError(f"Too complex expression {'.'.join(expression)}.")
                valueArray = getattr(self._extensions[extension], funcName)(*arguments)
            elif isinstance(expression, str):
                value = [system[expression] for system in self._pool]
                # value = [self.evaluateTerminal(expression, system) for system in self.pool]
                # unfortunately simple np.asarray spoils dictionaries
                if value and isinstance(value[0], Mapping):
                    valueArray = np.empty((len(value,)), dtype=type(value[0]))
                    for i, x in enumerate(value):
                        valueArray[i] = x
                else:
                    valueArray = np.asarray(value)
            else:
                # just a parameter. return it without doing anything.
                return expression
            self._storedData[expression] = valueArray
        return self._storedData[expression]

    def setAllExpressions(self) -> None:
        for expression, values in self._storedData.items():
            for s, value in zip(self._pool, values):
                s.setExpression(expression, value)
