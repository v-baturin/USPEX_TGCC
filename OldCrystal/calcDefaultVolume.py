"""
USPEX.Common.Atomistic.calcDefaultVolume.py
===========================================

Methods for volume calculation of a cell

.. codeauthor:: Maxim Rakitin <maxim.rakitin@gmail.com>
"""

import math

from ..Atomistic.Element import Element
from ..Atomistic.CompositionSpace import Composition


def VinetEOS(B0: float, B00: float, V0: float, x: float):
    """
    The function fits the Volume with the Vinet-EOS (equation of state).
    :Link: https://en.wikipedia.org/wiki/Rose%E2%80%93Vinet_equation_of_state

    :type B0: float
    :param B0: isothermal bulk modulus.
    :type B00: float
    :param B00: derivative of bulk modulus with respect to pressure.
    :type V0: float
    :param V0: volume at zero pressure.
    :type x: float
    :param x: volume at pressure P.
    :rtype: float
    :return: pressure
    """

    # B00 = B0'
    P = 3.0 * B0 * (1.0 - math.pow(x/V0, 1.0/3.0)) / \
        math.pow(x/V0, 2.0/3.0) * math.exp(1.5 * (B00-1.0) * (1.0-math.pow(x/V0, 1.0/3.0)))

    return P


def arange(start, end, step):
    """
    Create arange like in numpy (np.arange).

    :type start: float
    :param start: start value for the grid.
    :type end: float
    :param end: end value for the grid.
    :type step: float
    :param step: step to create the grid.
    :rtype: list
    :return: a list with values of the grid.
    """

    num_steps = int(round((end - start) / step))
    values_list = []
    for i in range(num_steps):
        value_i = start + float(i) * step
        values_list.append(value_i)

    if end - (value_i + step) >= 0.0001:
        value_i += step
        values_list.append(value_i)

    return values_list


def calcVolumePure(targetPress: float, atomType, systemType: str = 'atom'):    # Returns targetVolume
    """
    The function calculates a volume of a single element/molecule at the target pressure.

    :type targetPress: float
    :param targetPress: target pressure.
    :type atomType: str
    :param atomType: types of atoms;
    :type systemType: str
    :param systemType: molecular or atomic system.
    :rtype: float
    :return: volume of the element or molecule.
    """

    moleculeType = [1, 6, 7, 8, 9, 15, 16, 17, 33, 34, 35, 52, 53]

    if Element(atomType).z in moleculeType and systemType == 'mol':
        #             atomType   B0       B0'  V at 0GPa  V at 500GPa
        fitParameter = [
                        [  1,   8.937200,  4.77840,   7.0100,   1.1796],
                        [  6,  21.644000,  6.65530,  10.5800,   3.6400],
                        [  7, 229.770000,  3.34200,   7.0700,   3.4600],
                        [  8,   9.143400,  7.27150,  13.4800,   3.9300],
                        [  9,   1.981700,  8.33490,  18.3900,   4.0600],
                        [ 15,  90.811000,  3.64190,  16.2900,   5.7900],
                        [ 16,   4.477000,  6.60710,  29.4300,   6.0400],
                        [ 17,   3.897100,  6.46590,  34.7700,   6.5600],
                        [ 33,  53.119000,  4.85890,  22.4600,   7.9800],
                        [ 34,   6.894500,  7.18770,  31.4400,   8.3100],
                        [ 35,   4.534100,  7.03730,  37.5300,   8.4500],
                        [ 52,  10.385000,  6.97140,  38.4600,  11.1300],
                        [ 53,   2.819800,  7.63060,  50.3400,  10.8900],                       ]
        #for i in fitParameter:
        #    print '[%3i, %10.6f, %8.5f, %8.4f, %8.4f],' % (i[0], i[1], i[2], i[3], i[4])

    else:#elif systemType == 'atom':
        #               atomType   B0       B0'     V at 0GPa   V at 500GPa
        fitParameter = [
                        [  1,  95.351000,  4.25090,   2.8900,   1.1577],
                        [  2,   0.497530,  7.29940,  17.9700,   1.9200],
                        [  3,  15.083000,  3.79310,  20.1200,   3.1300],
                        [  4, 114.380000,  3.85940,   7.8400,   3.1700],
                        [  5, 234.190000,  3.89840,   6.1600,   3.2000],
                        [  6, 166.740000,  4.58850,   6.7300,   3.3600],
                        [  7, 184.700000,  4.17460,   7.2100,   3.5700],
                        [  8, 152.020000,  4.83250,   7.7500,   3.8500],
                        [  9,  55.289000,  5.77810,   9.9250,   4.0400],
                        [ 10,   2.285900,  7.78330,  20.2950,   4.2300],
                        [ 11,   5.661000,  5.11660,  36.6200,   5.5400],
                        [ 12,  28.819000,  4.92930,  22.8700,   6.5800],
                        [ 13,  62.769000,  4.81420,  17.0100,   6.3700],
                        [ 14,  78.151000,  4.93260,  14.8500,   6.0800],
                        [ 15,  79.673000,  4.99890,  14.3000,   5.9400],
                        [ 16,  84.499000,  4.16420,  15.7400,   5.9500],
                        [ 17,  38.801000,  4.69220,  21.2400,   6.5400],
                        [ 18,   1.378500,  7.31460,  47.2100,   7.4800],
                        [ 19,   0.054489,  9.84190,  73.5100,   7.3400],
                        [ 20,   2.455300,  6.57730,  42.2100,   6.9200],
                        [ 21,  33.001000,  4.27200,  24.7600,   6.5600],
                        [ 22, 147.000000,  1.96530,  16.9200,   5.2200],
                        [ 23, 222.720000,  2.78920,  13.2100,   5.9300],
                        [ 24, 282.200000,  3.93370,  11.3700,   6.2700],
                        [ 25, 289.360000,  4.49050,  10.6700,   6.1700],
                        [ 26, 272.910000,  4.74340,  10.4500,   6.0500],
                        [ 27, 234.760000,  5.13070,  10.5200,   6.0100],
                        [ 28, 198.310000,  5.22940,  10.8700,   5.9900],
                        [ 29, 126.910000,  5.70140,  12.0300,   6.1200],
                        [ 30,  58.971000,  6.02450,  15.4200,   6.5700],
                        [ 31,  43.538000,  5.94530,  19.1000,   7.3900],
                        [ 32,  56.040000,  5.70910,  19.2600,   7.8100],
                        [ 33,  70.008000,  5.41260,  18.9700,   7.9500],
                        [ 34,  68.629000,  5.10740,  20.2800,   8.1400],
                        [ 35,  28.715000,  5.51050,  26.6500,   8.4900],
                        [ 36,   1.475100,  7.24740,  57.8700,   9.2400],
                        [ 37,   0.054586,  9.94980,  90.4200,   9.2900],
                        [ 38,   1.239000,  7.82730,  53.1900,   9.1900],
                        [ 39,  14.711000,  5.95790,  32.9200,   9.0200],
                        [ 40, 156.990000,  4.37300,  18.0600,   8.6700],
                        [ 41, 108.850000,  2.01820,  22.8200,   5.9600],
                        [ 42, 281.340000,  4.00900,  15.5900,   8.6400],
                        [ 43, 307.230000,  4.45790,  14.6200,   8.5700],
                        [ 44, 283.740000,  4.87860,  14.2800,   8.4200],
                        [ 45, 230.820000,  5.36570,  14.5500,   8.4000],
                        [ 46, 163.210000,  5.83920,  15.4800,   8.4700],
                        [ 47,  81.667000,  6.45320,  18.0800,   8.7300],
                        [ 48,  35.645000,  6.75110,  23.1200,   9.2600],
                        [ 49,  28.333000,  6.25590,  28.0900,   9.9400],
                        [ 50,  39.875000,  6.09260,  27.4400,  10.5300],
                        [ 51,  48.231000,  5.84660,  27.2500,  10.7400],
                        [ 52,  47.471000,  5.61330,  28.4700,  10.8500],
                        [ 53,  20.759000,  5.89330,  35.8100,  10.8900],
                        [ 54,   0.798200,  7.96420,  78.6700,  12.0600],
                        [ 55,   0.038483, 10.36100, 116.1900,  11.8500],
                        [ 56,   0.886590,  8.86160,  63.5500,  12.1800],
                        [ 57,  10.752000,  6.71030,  37.5700,  10.5400],
                        [ 58,  33.765000,  6.46360,  26.3100,  10.0500],
                        [ 59,  37.211000,  7.30000,  22.8200,   9.7500],
                        [ 60,  45.533000,  7.56050,  20.7000,   9.4900],
                        [ 61,  45.533000,  7.56050,  20.7000,   9.4900],
                        [ 62,  45.533000,  7.56050,  20.7000,   9.4900],
                        [ 63,  45.533000,  7.56050,  20.7000,   9.4900],
                        [ 64,  45.533000,  7.56050,  20.7000,   9.4900],
                        [ 65,  11.801000,  6.70320,  32.7400,   9.4700],
                        [ 66,  12.543000,  6.58000,  32.2700,   9.3300],
                        [ 67,  14.335000,  6.32730,  31.7700,   9.2000],
                        [ 68,  16.160000,  6.09620,  31.2300,   9.0500],
                        [ 69,  16.160000,  6.09620,  31.2300,   9.0500],
                        [ 70,   7.710800,  6.14630,  37.5400,   8.4900],
                        [ 71,  17.962000,  5.88090,  30.7100,   8.8800],
                        [ 72, 126.780000,  2.41870,  22.0100,   7.0600],
                        [ 73, 202.100000,  3.58920,  18.1200,   8.6900],
                        [ 74, 314.740000,  4.19200,  15.9200,   9.2300],
                        [ 75, 369.260000,  4.54130,  15.0200,   9.2700],
                        [ 76, 361.680000,  4.95600,  14.7400,   9.2500],
                        [ 77, 307.150000,  5.21800,  15.0700,   9.2300],
                        [ 78, 237.930000,  5.66290,  15.8400,   9.3700],
                        [ 79, 134.910000,  6.20430,  18.1300,   9.7100],
                        [ 80,  13.303000,  8.07950,  28.9300,  10.3500],
                        [ 81,  26.004000,  5.93180,  31.2000,  10.2700],
                        [ 82,  30.124000,  6.24930,  31.8800,  11.4800],
                        [ 83,  41.332000,  5.86320,  31.8300,  12.0000],
                        [ 84,  11.467000,  6.22220,  45.7600,  12.0600],
                        [ 85,  11.467000,  6.22220,  45.7600,  12.0600],
                        [ 86,  11.467000,  6.22220,  45.7600,  12.0600],
                        [ 87,  11.467000,  6.22220,  45.7600,  12.0600],
                        [ 88,  11.467000,  6.22220,  45.7600,  12.0600],
                        [ 89,  11.467000,  6.22220,  45.7600,  12.0600],
                        [ 90,  34.478000,  5.84230,  32.5500,  11.5800],
                        [ 91,  69.983000,  5.84640,  24.7200,  10.8500],
                        [ 92, 116.670000,  6.05380,  20.1700,  10.3200],
                        [ 93, 168.740000,  6.19160,  17.6000,   9.9200],
                        [ 94, 197.720000,  6.36270,  16.4200,   9.6800],
                       ]
        #for i in fitParameter:
        #    print '[%3i, %10.6f, %8.5f, %8.4f, %8.4f],' % (i[0], i[1], i[2], i[3], i[4])
    #---------------------------------------------------------------------------

    for ii, parms in enumerate(fitParameter):
        if Element(parms[0]).short_name == atomType:
            i = ii
            break

    if targetPress > 500:
        volumeRange = arange(fitParameter[i][4]/2, fitParameter[i][4]+0.01, 0.02)
    else:
        volumeRange = arange(fitParameter[i][4], fitParameter[i][3], 0.05)

    tryVolume   = []
    tryPressure = []
    for tryVolume0 in volumeRange:
        tryVolume.append(tryVolume0)
        tmp = VinetEOS(fitParameter[i][2-1], fitParameter[i][3-1], fitParameter[i][4-1], tryVolume0)
        tryPressure.append(tmp)

    tmp_list = [abs(x-targetPress) for x in tryPressure] 
    whichOne = tmp_list.index(min(tmp_list))
    targetVolume = tryVolume[whichOne]
    
    return targetVolume


def calcVolume(targetPress: float, atomType, volumeType = 0):    # Returns targetVolume
    """
    The function calculates a volume of a single element/molecule at the target pressure.

    :type targetPress: float
    :param targetPress: target pressure.
    :type atomType: str
    :param atomType: types of atoms;
    :type volumeType: float
    :param volumeType:
        range 0 to 1, 0 corresponds to pure atomic environment for
        volume estimation, 1 to pure molecular one.
        Molecular environment is less dense.
        Intermediate value is a coefficient for molecular environment
        in linear combination of the two.
    :rtype: float
    :return: volume of the element or molecule.
    """

    return (     volumeType  * calcVolumePure(targetPress, atomType, 'atom') +
            (1 - volumeType) * calcVolumePure(targetPress, atomType, 'mol')   )

def calcVolumeForComposition(composition: Composition, externalPressure = 0.0001, volumeType = 0, **kwargs):
    """
    The function calculates a volume of the given composition at the target pressure.
    :type composition: Composition
    :param composition: Composition for which the volume is to be estimated.
    :type externalPressure: float
    :param externalPressure: External pressure for this structure in GPa.
    :type volumeType: float
    :param volumeType:
        range 0 to 1, 0 corresponds to pure atomic environment for
        volume estimation, 1 to pure molecular one.
        Molecular environment is less dense.
        Intermediate value is a coefficient for molecular environment
        in linear combination of the two.
    :rtype: float
    :return: volume
    """
    return sum(calcVolume(externalPressure, symbol, volumeType) * amount
               for symbol, amount in composition.elementalComposition.items())
