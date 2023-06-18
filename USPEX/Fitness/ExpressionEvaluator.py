"""
USPEX.Common.ExpressionEvaluator
================================

Data type representing rules of how we determine which systems are better.

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import logging
import numpy as np
from collections.abc import Mapping

from .presets import applyPresets


logger = logging.getLogger(__name__)


class ExpressionEvaluator:

    def __init__(self, pool, extensions):
        self.pool = pool
        self.extensions = extensions
        self._storedData = {}

    def evaluate(self, expression):
        expression = applyPresets(expression)
        if expression not in self._storedData:
            if len(self.pool) == 0:
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
                valueArray = getattr(self.extensions[extension], funcName)(*arguments)
            elif isinstance(expression, str):
                value = [self.evaluateTerminal(expression, system) for system in self.pool]
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

    def evaluateTerminal(self, expression, system):
        if expression in system:
            value = system[expression]
        else:
            extension, suffix, *extra = expression.split('.')
            assert not extra, f"Too complex property {expression}."
            value = getattr(self.extensions[extension], suffix)(system)
        return value

    def setAllExpressions(self):
        for expression, values in self._storedData.items():
            for s, value in zip(self.pool, values):
                s.setExpression(expression, value)

    @staticmethod
    def calculate(expression, pool, extensions):
        calculator = ExpressionEvaluator(pool, extensions)
        calculator.evaluate(expression)
        calculator.setAllExpressions()
