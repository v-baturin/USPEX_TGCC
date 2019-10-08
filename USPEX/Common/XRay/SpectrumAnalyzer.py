"""
@file:      SpectrumAnalyzer.py
@author:    Michele Galasso
@contact:   michele.galasso@skoltech.ru
@date:      4 October 2019
@brief:     Class which implements the fitness function for X-ray optimization.
"""

import numpy as np

from pymatgen.core.structure import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator


FACTORS = [
    5,              # fitness factor for I > 90
    1,              # fitness factor for 50 < I <= 90
    0.25,           # fitness factor for 10 < I <= 50
    0.02,           # fitness factor for 1 < I <= 10
    0               # fitness factor for I <= 1
]


class SpectrumAnalyzer(object):
    def __init__(self, spectrum_starts: int, spectrum_ends: int, wavelength: float, match_tol: float,
                 exp_angles: list, exp_intensities: list):
        """
        Initializes the class by parsing a file which contains information about the experimental spectrum
        :param experimentalSpectrum: file name
        """
        self.spectrum_starts = spectrum_starts
        self.spectrum_ends = spectrum_ends
        self.wavelength = wavelength
        self.match_tol = match_tol
        self.exp_angles = exp_angles
        self.exp_intensities = exp_intensities

    def __getitem__(self, item):
        """
        Special method which allows '[]' operator. Calculates the agreement with the experimental X-ray spectrum,
        """
        # compute agreement
        amplitude = self.spectrum_ends - self.spectrum_starts
        calculator = XRDCalculator(wavelength=self.wavelength)

        tmp = Structure(lattice=item.cell, species=item.get_chemical_symbols(), coords=item.scaled_coordinates)
        string = tmp.to(fmt='cif', symprec=0.2)
        structure = Structure.from_str(string, fmt='cif')

        # pure hydrogen gets low agreement
        if list(item.composition.keys()) == ['H']:
            return 100.0

        fitness = 0
        pattern = calculator.get_pattern(structure)
        th_angles = pattern.x[np.logical_and(pattern.x < self.spectrum_ends, pattern.x > self.spectrum_starts)]
        th_intensities = pattern.y[np.logical_and(pattern.x < self.spectrum_ends, pattern.x > self.spectrum_starts)]

        # match corresponding peaks
        exp_matches, th_matches = [], []
        for exp_index, (exp_angle, exp_intensity) in enumerate(zip(self.exp_angles, self.exp_intensities)):
            partial = 0
            counter = 0
            for th_index, (th_angle, th_intensity) in enumerate(zip(th_angles, th_intensities)):
                if np.abs(th_angle - exp_angle) < self.match_tol and th_intensity > 1:
                    counter += 1
                    exp_matches.append(exp_index)
                    th_matches.append(th_index)
                    factor = self.choose_factor(exp_intensity, FACTORS)
                    partial += factor * np.abs(exp_angle - th_angle) ** 2 / amplitude ** 2
                    partial += factor * np.abs(exp_intensity - th_intensity) ** 2 / 100 ** 2
            # average out in the case when multiple theoretical peaks match to the same experimental peak
            if partial:
                fitness += partial / counter

        exp_angles_rest = np.delete(self.exp_angles, exp_matches)
        exp_intensities_rest = np.delete(self.exp_intensities, exp_matches)
        th_angles_rest = np.delete(th_angles, th_matches)
        th_intensities_rest = np.delete(th_intensities, th_matches)

        # experimental rest
        for angle, intensity in zip(exp_angles_rest, exp_intensities_rest):
            factor = self.choose_factor(intensity, FACTORS)
            fitness += factor * angle ** 2 / amplitude ** 2
            fitness += factor * intensity ** 2 / 100 ** 2

        # theoretical rest
        for angle, intensity in zip(th_angles_rest, th_intensities_rest):
            factor = self.choose_factor(intensity, FACTORS)
            fitness += factor * angle ** 2 / amplitude ** 2
            fitness += factor * intensity ** 2 / 100 ** 2

        return fitness

    @staticmethod
    def parse(filename : str):
        """
        It parses a file containing information about the experimental spectrum
        and returns a dictionary containing the parameters for class initialization.
        :param filename:
        :return: dict
        """
        dct = {}
        with open(filename, 'r') as f:
            current_line = f.readline()
            while current_line != '':
                values = current_line.split()
                if values[0] == 'start':
                    dct['spectrum_starts'] = int(values[1])
                elif values[0] == 'end':
                    dct['spectrum_ends'] = int(values[1])
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

    @staticmethod
    def choose_factor(intensity : float, choices : list):
        """
        Simple auxiliary function which selects a factor based on the value of intensity.
        :param intensity:
        :param choices:
        :return:
        """
        if intensity > 90:
            factor = choices[0]
        elif 50 < intensity <= 90:
            factor = choices[1]
        elif 10 < intensity <= 50:
            factor = choices[2]
        elif 1 < intensity <= 10:
            factor = choices[3]
        else:
            factor = choices[4]
        return factor
