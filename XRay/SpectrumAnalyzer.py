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


class SpectrumAnalyzer(object):
    def __init__(self, spectrum_starts: float, spectrum_ends: float, wavelength: float, match_tol: float,
                 exp_angles: list, exp_intensities: list):
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
        """
        self.k = None

        self.spectrum_starts = spectrum_starts
        self.spectrum_ends = spectrum_ends
        self.wavelength = wavelength
        self.match_tol = match_tol
        self.exp_angles = np.array(exp_angles)
        self.exp_intensities = np.array(exp_intensities) / max(exp_intensities) * 100

    def __call__(self, system):
        """
        Special method which allows the '[]' operator. Calculates the agreement with the experimental X-ray spectrum.

        :type system: :class:`AtomicStructure` or one of its subclasses
        :param system: the crystal structure whose theoretical spectrum needs to be compared with experiment.
        :rtype: float
        :return: a fitness denoting how much the theoretical spectrum differs from the experiment.
        """
        # pure hydrogen gets low agreement
        if list(system.composition.keys()) == ['H']:
            return 100.0

        # symmetrize the candidate structure
        tmp = Structure(lattice=system.cell, species=system.get_chemical_symbols(), coords=system.scaled_coordinates)
        string = tmp.to(fmt='cif', symprec=0.2)
        structure = Structure.from_str(string, fmt='cif')

        # initialize the lattice factor k
        params = np.array([1.0])

        # compute the lattice factor which minimize fitness
        result = minimize(self.fitness_function, params, (structure,), method='Powell', bounds=((0.98, 1.02),),
                          options={'ftol': 1.0e-4})

        if not result.success:
            raise RuntimeError('Scipy minimize could not calculate the agreement with experimental X-ray data.')

        # extract the lattice factor
        self.k = result.x[0]

        # return fitness
        return result.fun

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
