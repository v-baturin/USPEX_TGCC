import logging

logFormatter = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename='Client.log', format=logFormatter, level=logging.INFO)

import asyncio

from .components import read, compileParams, GenerationController

input = read('input.uspex')
params = compileParams(**input)
controller = GenerationController.createController(**params)
asyncio.get_event_loop().run_until_complete(controller.run())
