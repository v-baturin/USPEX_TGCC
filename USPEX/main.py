import logging
import sys
import asyncio

logging.basicConfig(filename='log', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='USPEX')

    parser.add_argument("-v", "--version", dest="version", action="store_true",
                        help="show program's version number and exit")

    # parser.add_argument("-p", "--parameter", action='callback', dest="parm",
    #                     help="specify parameter to get help. If no value or 'all' value is specified, all INPUT.txt parameters will be shown",
    #                     metavar="PARM")

    # parser.add_argument("-p", "--parameter", action='callback', callback=optional_arg('all'), dest="parm",
    #                     help="specify parameter to get help. If no value or 'all' value is specified, all INPUT.txt parameters will be shown",
    #                     metavar="PARM")

    # parser.add_argument("-e", "--example", dest="example",
    #                     help="show USPEX example details. If no value or 'all' value is specified, all examples will be shown")

    # parser.add_argument("-e", "--example", action='callback', callback=optional_arg('all'), dest="example",
    #                     help="show USPEX example details. If no value or 'all' value is specified, all examples will be shown",
    #                     metavar="NUM")


    # parser.add_argument("-c", "--copy", dest="copyID",
    #                     help="copy the INPUT.txt file and Specific Folder of ExampleXX.")

    # parser.add_argument("-g", "--generate", dest="generate", action="store_true",
    #                     help="generate directories for preparing an USPEX calculation, including AntiSeeds, Seeds, Specific, Submission folders")

    parser.add_argument("-r", "--run", dest="uspex_run", action="store_true",
                      help="run USPEX calculation")

    parser.add_argument("--clean", "-c", dest="uspex_clean", action="store_true",
                        help="clean calculation folder")

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()

    if args.version:
        from . import __version__
        print(f'USPEX {__version__}\n')

    # if args.parm:
    #     parm = args.parm
    #     uspex_help(parm)
    #     sys.exit(0)

    # if args.example:
    #     from uspex_examples import uspex_examples
    #     uspex_examples(args.example)
    #     sys.exit(0)

    # Copy
    # if args.copyID:
    #     from uspex_examples import uspex_copy
    #     uspex_copy(args.copyID)
    #     sys.exit(0)

    if args.uspex_clean:
        from os import listdir, getcwd, path, remove
        cwd = getcwd()
        for file in listdir(cwd):
            if path.isfile(path.join(cwd, file)) and file in ['controller.dump',
                                                              'controller.dump.back',
                                                              'stages.dump',
                                                              'stages.dump.back',
                                                              'USPEX_IS_DONE']:
                remove(path.join(cwd, file))

    # if args.generate:
    #     # Copy Submission dir from USPEXPATH to run dir:
    #     copy_submission()
    #
    #     gen_dirs = [
    #         'AntiSeeds',
    #         'Seeds',
    #         'Specific',
    #     ]
    #     for gen_dir in gen_dirs:
    #         if not os.path.isdir(gen_dir):
    #             os.mkdir(gen_dir)
    #             print('\t' + gen_dir + ' dir was created.')
    #         else:
    #             print('\t' + gen_dir + ' dir already exists.')
    #     sys.exit(0)

    # -------------------------------------------------------------------------------


    if args.uspex_run:
        _run()

def _run():
    try:
        from .components import GenerationController
        asyncio.get_event_loop().run_until_complete(GenerationController.createController().run())
    except Exception as ex:
        logger.exception(ex)
        exc_info = sys.exc_info()
        raise exc_info[0].with_traceback(exc_info[1], exc_info[2])


if __name__ == '__main__':
    _run()