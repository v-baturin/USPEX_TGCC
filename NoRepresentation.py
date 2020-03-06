class NoRepresentation(object):


    def __init__(self, name : str, selection : str, target : str, **kwargs):
        pass

    def presentSystems(self, systems, numStages, fitness):
        pass

    def presentOutput(self, targetConfig, selectionConfig, numStages, numParallelCalcs, populations, fitness):
        pass

    def presentAnalysis(self, analyses):
        pass

    def presentPool(self, pools, fitness):
        pass
