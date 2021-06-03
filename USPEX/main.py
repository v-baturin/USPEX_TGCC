import logging

logFormatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename='Client.log', format=logFormatter, level=logging.INFO)

import asyncio

from .components import GenerationController

asyncio.get_event_loop().run_until_complete(GenerationController.createController().run())
