"""
USPEX.Common.XRay.SpectrumAnalyzer
==================================

Class which implements the fitness function for comparing X-ray spectra

.. codeauthor:: Michele Galasso <m.galasso@yandex.com>
"""

import numpy as np

from copy import deepcopy
from scipy.optimize import minimize
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator

from .get_reflections import get_reflections


class SpectrumAnalyzer(object):
    def __init__(self, spectrum_starts: float = None, spectrum_ends: float = None, wavelength: float = None,
                 match_tol: float = None, exp_angles: list = None, exp_intensities: list = None,
                 exp_reflections: list = None):
        """
        Initializes the class.

        :type spectrum_starts: float
        :param spectrum_starts:
            minimum diffraction angle considered for the comparison between theoretical and experimental spectra.
        :type spectrum_ends: float
        :param spectrum_ends:
            maximum diffraction angle considered for the comparison between theoretical and experimental spectra.
        :type wavelength: float
        :param wavelength: experimental wavelength.
        :type match_tol: float
        :param match_tol: tolerance in degrees for considering a match between two peaks.
        :type exp_angles: list[float]
        :param exp_angles: angles of the experimental spectrum.
        :type exp_intensities: list[float]
        :param exp_intensities: intensities of the experimental spectrum.
        :type exp_reflections: list[float, tuple(int, int, int), float]
        :param exp_reflections: single crystal experimental reflections.
        """
        if exp_reflections is not None:
            self.mode = 'scxrd'
            self.exp_reflections = np.array(exp_reflections)
        else:
            self.mode = 'powder'
            self.spectrum_starts = spectrum_starts
            self.spectrum_ends = spectrum_ends
            self.wavelength = wavelength
            self.match_tol = match_tol
            self.exp_angles = np.array(exp_angles)
            self.exp_intensities = np.array(exp_intensities) / max(exp_intensities) * 100

    def analyze(self, system):
        """
        Special method which allows the '[]' operator. Calculates the agreement with the experimental X-ray spectrum.

        :type system: :class:`AtomicStructure` or one of its subclasses
        :param system: the crystal structure whose theoretical spectrum needs to be compared with experiment.
        :rtype: float
        :return: a fitness denoting how much the theoretical spectrum differs from the experiment.
        """
        structure = system['structure']

        # pure hydrogen gets low agreement
        if list(structure.composition.keys()) == ['H']:
            return 100.0

        # create a pymatgen Structure object
        structure = Structure(lattice=structure.cell, species=structure.get_chemical_symbols(),
                              coords=structure.scaled_coordinates)

        if self.mode == 'powder':
            # symmetrize the candidate structure
            cif_string = structure.to(fmt='cif', symprec=0.2)
            structure = Structure.from_str(cif_string, fmt='cif')

            # initialize the lattice factor k
            params = np.array([1.0])

            # compute the lattice factor which minimize fitness
            result = minimize(self.fitness_function, params, (structure,), method='Powell', bounds=((0.98, 1.02),),
                              options={'ftol': 1.0e-4})

            if not result.success:
                raise RuntimeError('Scipy minimize could not calculate the agreement with experimental X-ray data.')

            system['spectrumAnalyzer.xraydistance'] = result.fun
            system['spectrumAnalyzer.k'] = result.x[0]
        else:
            # compute minimum d spacing
            min_d_spacing = 10000
            for reflection in self.exp_reflections:
                hkl = list(reflection[1])
                d_spacing = structure.lattice.d_hkl(hkl)
                if d_spacing < min_d_spacing:
                    min_d_spacing = d_spacing

            min_d_spacing = np.floor(min_d_spacing * 1000) / 1000
            th_reflections = get_reflections(structure, min_d_spacing)
            th_reflections = np.array(th_reflections)

            # scale theoretical intensities according to experimental maximum
            th_reflections[:, 0] = th_reflections[:, 0] / max(th_reflections[:, 0]) * max(self.exp_reflections[:, 0])

            # compute R-factor
            numerator = 0
            denominator = 0
            for i_hkl, hkl, sigma_hkl in self.exp_reflections:
                th_hkls = [r[1] for r in th_reflections]

                if hkl in th_hkls:
                    index = th_hkls.index(hkl)
                elif (-hkl[0], -hkl[1], -hkl[2]) in th_hkls:
                    index = th_hkls.index((-hkl[0], -hkl[1], -hkl[2]))
                else:
                    raise ValueError(f'Experimental reflection {hkl} with intensity {i_hkl} not found in theory.')

                numerator += (1 / sigma_hkl ** 2) * (i_hkl - th_reflections[index][0]) ** 2
                denominator += (1 / sigma_hkl ** 2) * i_hkl ** 2

            wR = np.sqrt(numerator / denominator)       # weighted R-factor

            system['spectrumAnalyzer.xraydistance'] = wR

    def k(self, system):
        if 'spectrumAnalyzer.k' not in system:
            self.analyze(system)
        assert 'spectrumAnalyzer.k' in system
        return system['spectrumAnalyzer.k']

    def xraydistance(self, system):
        if 'spectrumAnalyzer.xraydistance' not in system:
            self.analyze(system)
        assert 'spectrumAnalyzer.xraydistance' in system
        return system['spectrumAnalyzer.xraydistance']

    @staticmethod
    def parse(filename: str):
        """
        It parses a file containing information about the experimental spectrum and
        it returns a dictionary containing the parameters for class initialization.

        :type filename: str
        :param filename: filename.
        :rtype: dict
        :return: kwargs parsed from the file.
        """
        dct = {}
        with open(filename, 'r') as f:
            if filename.endswith('.hkl'):
                dct['exp_reflections'] = []
                for line in f:
                    values = line.split()

                    # the hkl file terminates with all zeros
                    if values == ['0', '0', '0', '0.00', '0.00']:
                        break

                    i_hkl = float(values[3])
                    if i_hkl > 0:
                        hkl = (int(values[0]), int(values[1]), int(values[2]))
                        sigma_hkl = float(values[4])
                        dct['exp_reflections'].append([i_hkl, hkl, sigma_hkl])
            else:
                current_line = f.readline()
                while current_line != '':
                    values = current_line.split()
                    if values[0] == 'start':
                        dct['spectrum_starts'] = float(values[1])
                    elif values[0] == 'end':
                        dct['spectrum_ends'] = float(values[1])
                    elif values[0] == 'wavelength':
                        dct['wavelength'] = float(values[1])
                    elif values[0] == 'match_tol':
                        dct['match_tol'] = float(values[1])
                    elif values[0] == 'peaks':
                        dct['exp_angles'] = [float(values[1])]
                        dct['exp_intensities'] = [float(values[2])]
                        current_line = f.readline()
                        while current_line != '':
                            values = current_line.split()
                            dct['exp_angles'].append(float(values[0]))
                            dct['exp_intensities'].append(float(values[1]))
                            current_line = f.readline()
                    current_line = f.readline()
        return dct

    def fitness_function(self, params, structure):
        # initialize fitness
        fitness = 0

        # initialize XRDCalculator
        calculator = XRDCalculator(wavelength=self.wavelength)

        # apply lattice correction and get pattern
        k = params[0]
        candidate = deepcopy(structure)
        candidate.lattice = Lattice(np.diag([k, k, k]) @ candidate.lattice.matrix)
        pattern = calculator.get_pattern(candidate, two_theta_range=(self.spectrum_starts, self.spectrum_ends))
        th_angles = pattern.x
        th_intensities = pattern.y

        # match corresponding peaks
        exp_matches, th_matches = [], []
        for exp_index, (exp_angle, exp_intensity) in enumerate(zip(self.exp_angles, self.exp_intensities)):
            partial = 0
            counter = 0
            for th_index, (th_angle, th_intensity) in enumerate(zip(th_angles, th_intensities)):
                if np.abs(th_angle - exp_angle) < self.match_tol:
                    counter += 1
                    exp_matches.append(exp_index)
                    th_matches.append(th_index)
                    partial += ((exp_intensity - th_intensity) / 100) ** 2 * (exp_intensity / 100) ** 2
            # average out in the case when multiple theoretical peaks match to the same experimental peak
            if partial:
                fitness += partial / counter

        exp_intensities_rest = np.delete(self.exp_intensities, exp_matches)
        th_intensities_rest = np.delete(th_intensities, th_matches)

        # experimental rest
        for intensity in exp_intensities_rest:
            fitness += (intensity / 100) ** 2

        # theoretical rest
        for intensity in th_intensities_rest:
            fitness += (intensity / 100) ** 2

        return fitness
