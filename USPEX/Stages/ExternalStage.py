import logging


logger = logging.getLogger(__name__)


class ExternalStage:

    executorType = None

    @classmethod
    def registerTypes(cls, executorType):
        cls.executorType = executorType

    def __init__(self, tag, source, **kwargs):
        self.tag = tag
        self.source = source
        self.executor = self.executorType(tag=tag, **kwargs)

    async def run(self, system):
        try:
            result = await self.executor.run(system.ID, system.getFlavour(self.source))
        except Exception as ex:
            logger.warning(f'system {system.ID} error in relaxation:')
            logger.exception(ex)
            system.setProperty('isBad', True, suffix=self.tag)
            return
        system.addFlavour(self.tag, result)
        system.setProperty('isBad', False, suffix=self.tag)

