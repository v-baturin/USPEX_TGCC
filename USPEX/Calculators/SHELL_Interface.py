class SHELL_Interface(object):
    inputFile, outputFile, errorFile = 'input', 'output', 'error'

    _DEFAULT_SLEEP_TIME = 1

    def __init__(self, sleepTime : int = None, **kwargs):
        if sleepTime is not None and sleepTime > 0:
            self.sleepTime = sleepTime
        else:
            self.sleepTime = self._DEFAULT_SLEEP_TIME

    def isConverged(self, calcFolder : str):
        return True

    def readOutput(self, system, calcFolder : str):
        pass

    def prepareLocalCalculation(self, system, calcFolder : str):
        pass
