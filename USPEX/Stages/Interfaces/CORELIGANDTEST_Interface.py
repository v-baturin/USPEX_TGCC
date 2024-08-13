"""
USPEX.Stages.CORELIGANDTEST_Interface
============================

.. codeauthor:: Vladimir Baturin <vsbat@yandex.ru>

"""
import logging
import numpy as np
import yaml
from functions_collection import GO_testing_function, function_lib

from pathlib import Path

logger = logging.getLogger(__name__)


class CORELIGANDTEST_Interface:
    """
     Fake calculator for tests Gulp.
     Local running
    """
    inputFile, outputFile, errorFile = 'input', 'output', 'error'

    DEFAULT_SLEEP_TIME = 1

    def __init__(self, tag: str, interactionSetup: str | None = None, targetProperties: list = None, **kwargs):
        """

        :param params: dictionary with parameters:
                * mop_input: (str) path to ginput-file.
                * vacuumSize=10
        """

        self.tag = tag

        interactionSetup = Path.cwd() / f'Specific/test_setup.yaml' if interactionSetup is None else Path(
            interactionSetup)
        assert interactionSetup.exists()

        with open(interactionSetup, 'r') as interacions_fid:
            interactions_list = yaml.safe_load(interacions_fid.read())['interactions']
            self.interactions = dict()
            for int_dict in interactions_list:
                fn = GO_testing_function(**function_lib[int_dict['function']])
                fn.transform_to_match_new_borders(((0., 2 * np.pi), (0., np.pi)))
                self.interactions[frozenset(int_dict['elements'].split())] = {'function': fn}
                if 'optVertex' in int_dict:
                    self.interactions[frozenset(int_dict['elements'].split())]['optVertex'] = int_dict['optVertex']

        self.targetProperties = targetProperties

        logger.debug('CORELIGANDTEST calculator created.')

    def prepareLocalCalculation(self, system, calcFolder: Path):
        """

        :param system:
        :param calcFolder:
        """

        with open(calcFolder / self.inputFile, 'wt') as f:
            f.write('')

        logger.debug('CORELIGANDTEST calculator prepared calculation.')
        return ''

    def isConverged(self, calcFolder: Path):
        """
        :param calcFolder:
        :return: whether optimization converged
        """

        return True

    def readOutput(self, system, calcFolder: Path):
        factory = system.getFactory()
        result = factory()
        fake_energy = 0.
        env = system.getProperty('environments', extension='atomistic')[0][0]
        centered_env_coords = env.getCartesianCoordinates()
        centered_env_coords -= np.sum(centered_env_coords, axis=0) / len(centered_env_coords)
        for k, v in self.interactions.items():
            if 'optVertex' in v:
                opt_idx = v['optVertex']
                opt_vertex_coordinates = centered_env_coords[opt_idx]
                phi1, theta1 = phi(opt_vertex_coordinates), theta(opt_vertex_coordinates)
                v['function'].affine_transform(b=(v['function'].global_optima[0][0][0] - phi1,
                                                  v['function'].global_optima[0][0][1] - theta1))

        for mol in system.getProperty('molecules', extension='atomistic'):
            molCoord = mol.getCartesianCoordinates()[0]
            distances = np.linalg.norm(env.getCartesianCoordinates() - molCoord, axis=1)
            closest_index = np.argmin(distances)
            closest_vertex = centered_env_coords[closest_index]
            phi_closest, theta_closest = phi(closest_vertex), theta(closest_vertex)
            mol_atom = mol.getAtomTypes()[0].short_name
            closest_env_atom = env.getAtomTypes()[closest_index].short_name
            ads_en = self.interactions[frozenset((mol_atom, closest_env_atom))]['function'](phi_closest, theta_closest)
            fake_energy += ads_en
        result.setProperty('enthalpy', ads_en)
        result.setProperty('isBad', False)
        return result


def theta(vec):
    x, y, z = vec
    return np.arccos(z / np.linalg.norm([x, y, z]))


def phi(vec):
    x, y, z = vec
    return np.sign(y) * np.arccos(x / np.linalg.norm([x, y]))
