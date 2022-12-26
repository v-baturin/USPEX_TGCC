class Stages:

    knownStages = {}

    @classmethod
    def registerStage(cls, name, stageType: type):
        assert name not in cls.knownStages
        cls.knownStages[name] = stageType

    @classmethod
    def createStage(cls, stageType, **kwargs):
        return cls.knownStages[stageType](**kwargs)