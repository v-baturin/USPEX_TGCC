"""
USPEX.Common.Atomistic.Element
==============================

Class for Element

.. codeauthor:: Maxim Rakitin
.. codeauthor:: Artem Samtsevich <samtsevichartem@gmail.com>
"""

from typing import List, Union, Optional


class _Atom:
    z: int                         # atomic number
    shortname: str                 # shortname
    fullname: str                  # Full name of the element
    valence: float                 # Valence of the element
    v_electrons: int               # number of valence electrons
    R_covalent: Optional[float]    # covalent radius of the element
    good_bonds: float              # good bonds
    mass: float                    # element mass

    def __init__(self, z: int, shortname: str, fullname: str, valence: float, v_electrons: int,
                 R_covalent: Optional[float], good_bonds: float, mass: float):
        self.z = z
        self.shortname = shortname
        self.fullname = fullname
        self.valence = valence
        self.v_electrons = v_electrons
        self.R_covalent = R_covalent
        self.good_bonds = good_bonds
        self.mass = mass


_ELEMENTS_LIST = [
    _Atom(1, 'H', 'Hydrogen', 1.0, 1, 0.31, 0.2, 1.007940),
    _Atom(2, 'He', 'Helium', 0.5, 2, 0.28, 0.05, 4.002602),
    _Atom(3, 'Li', 'Lithium', 1.0, 1, 1.28, 0.1, 6.941000),
    _Atom(4, 'Be', 'Beryllium', 2.0, 2, 0.96, 0.2, 9.012182),
    _Atom(5, 'B', 'Boron', 3.0, 3, 0.84, 0.3, 10.811000),
    _Atom(6, 'C', 'Carbon', 4.0, 4, 0.76, 0.5, 12.010700),
    _Atom(7, 'N', 'Nitrogen', 3.0, 5, 0.71, 0.5, 14.006700),
    _Atom(8, 'O', 'Oxygen', 2.0, 6, 0.66, 0.3, 15.999400),
    _Atom(9, 'F', 'Fluorine', 1.0, 7, 0.57, 0.1, 18.998400),
    _Atom(10, 'Ne', 'Neon', 0.5, 8, 0.58, 0.05, 20.179700),
    _Atom(11, 'Na', 'Sodium', 1.0, 1, 1.66, 0.05, 22.989770),
    _Atom(12, 'Mg', 'Magnesium', 2.0, 2, 1.41, 0.1, 24.305000),
    _Atom(13, 'Al', 'Aluminium', 3.0, 3, 1.21, 0.2, 26.981540),
    _Atom(14, 'Si', 'Silicon', 4.0, 4, 1.11, 0.3, 28.085500),
    _Atom(15, 'P', 'Phosphorus', 3.0, 5, 1.07, 0.3, 30.973760),
    _Atom(16, 'S', 'Sulfur', 2.0, 6, 1.05, 0.2, 32.065000),
    _Atom(17, 'Cl', 'Chlorine', 1.0, 7, 1.02, 0.1, 35.453000),
    _Atom(18, 'Ar', 'Argon', 0.5, 8, 1.06, 0.05, 39.948000),
    _Atom(19, 'K', 'Potassium', 1.0, 1, 2.03, 0.05, 39.098300),
    _Atom(20, 'Ca', 'Calcium', 2.0, 2, 1.76, 0.1, 40.078000),
    _Atom(21, 'Sc', 'Scandium', 3.0, 3, 1.7, 0.2, 44.955910),
    _Atom(22, 'Ti', 'Titanium', 4.0, 4, 1.6, 0.3, 47.867000),
    _Atom(23, 'V', 'Vanadium', 4.0, 5, 1.53, 0.3, 50.941500),
    _Atom(24, 'Cr', 'Chromium', 3.0, 6, 1.39, 0.25, 51.996100),
    _Atom(25, 'Mn', 'Manganese', 4.0, 5, 1.39, 0.3, 54.938050),
    _Atom(26, 'Fe', 'Iron', 3.0, 3, 1.32, 0.25, 55.845000),
    _Atom(27, 'Co', 'Cobalt', 3.0, 3, 1.26, 0.25, 58.933200),
    _Atom(28, 'Ni', 'Nickel', 2.0, 3, 1.24, 0.15, 58.693200),
    _Atom(29, 'Cu', 'Copper', 2.0, 2, 1.32, 0.1, 63.546000),
    _Atom(30, 'Zn', 'Zinc', 2.0, 2, 1.22, 0.1, 65.409000),
    _Atom(31, 'Ga', 'Gallium', 3.0, 3, 1.22, 0.25, 69.723000),
    _Atom(32, 'Ge', 'Germanium', 4.0, 4, 1.2, 0.5, 72.640000),
    _Atom(33, 'As', 'Arsenic', 3.0, 5, 1.19, 0.35, 74.921600),
    _Atom(34, 'Se', 'Selenium', 2.0, 6, 1.2, 0.2, 78.960000),
    _Atom(35, 'Br', 'Bromine', 1.0, 7, 1.2, 0.1, 79.904000),
    _Atom(36, 'Kr', 'Krypton', 0.5, 8, 1.16, 0.05, 83.798000),
    _Atom(37, 'Rb', 'Rubidium', 1.0, 1, 2.2, 0.05, 86.467800),
    _Atom(38, 'Sr', 'Strontium', 2.0, 2, 1.95, 0.1, 87.620000),
    _Atom(39, 'Y', 'Yttrium', 3.0, 3, 1.9, 0.2, 88.905850),
    _Atom(40, 'Zr', 'Zirconium', 4.0, 4, 1.75, 0.3, 91.224000),
    _Atom(41, 'Nb', 'Niobium', 5.0, 5, 1.64, 0.35, 92.906380),
    _Atom(42, 'Mo', 'Molybdenum', 4.0, 6, 1.54, 0.3, 95.940000),
    _Atom(43, 'Tc', 'Technetium', 4.0, 5, 1.47, 0.3, 98.000000),
    _Atom(44, 'Ru', 'Ruthenium', 4.0, 3, 1.46, 0.3, 101.070000),
    _Atom(45, 'Rh', 'Rhodium', 4.0, 3, 1.42, 0.3, 102.905500),
    _Atom(46, 'Pd', 'Palladium', 4.0, 3, 1.39, 0.3, 106.420000),
    _Atom(47, 'Ag', 'Silver', 1.0, 2, 1.45, 0.05, 107.868200),
    _Atom(48, 'Cd', 'Cadmium', 2.0, 2, 1.44, 0.1, 112.411000),
    _Atom(49, 'In', 'Indium', 3.0, 3, 1.42, 0.2, 114.818000),
    _Atom(50, 'Sn', 'Tin', 4.0, 4, 1.39, 0.3, 118.710000),
    _Atom(51, 'Sb', 'Antimony', 3.0, 5, 1.39, 0.2, 121.760000),
    _Atom(52, 'Te', 'Tellurium', 2.0, 6, 1.38, 0.2, 127.600000),
    _Atom(53, 'I', 'Iodine', 1.0, 7, 1.39, 0.1, 126.904500),
    _Atom(54, 'Xe', 'Xenon', 0.5, 8, 1.4, 0.05, 131.293000),
    _Atom(55, 'Cs', 'Caesium', 1.0, 1, 2.44, 0.05, 132.905500),
    _Atom(56, 'Ba', 'Barium', 2.0, 2, 2.15, 0.1, 137.327000),
    _Atom(57, 'La', 'Lanthanum', 3.0, 3, 2.07, 0.2, 138.905500),
    _Atom(58, 'Ce', 'Cerium', 4.0, 3, 2.04, 0.3, 140.116000),
    _Atom(59, 'Pr', 'Praseodymium', 3.0, 3, 2.03, 0.2, 140.907700),
    _Atom(60, 'Nd', 'Neodymium', 3.0, 3, 2.01, 0.2, 144.240000),
    _Atom(61, 'Pm', 'Promethium', 3.0, 3, 1.99, 0.2, 145.000000),
    _Atom(62, 'Sm', 'Samarium', 3.0, 3, 1.98, 0.2, 150.360000),
    _Atom(63, 'Eu', 'Europium', 3.0, 3, 1.98, 0.2, 151.964000),
    _Atom(64, 'Gd', 'Gadolinium', 3.0, 3, 1.96, 0.2, 157.250000),
    _Atom(65, 'Tb', 'Terbium', 3.0, 3, 1.94, 0.2, 158.925300),
    _Atom(66, 'Dy', 'Dysprosium', 3.0, 3, 1.92, 0.2, 162.500000),
    _Atom(67, 'Ho', 'Holmium', 3.0, 3, 1.92, 0.2, 164.930300),
    _Atom(68, 'Er', 'Erbium', 3.0, 3, 1.89, 0.2, 167.259000),
    _Atom(69, 'Tm', 'Thulium', 3.0, 3, 1.9, 0.2, 168.934200),
    _Atom(70, 'Yb', 'Ytterbium', 3.0, 3, 1.87, 0.2, 173.040000),
    _Atom(71, 'Lu', 'Lutetium', 3.0, 3, 1.87, 0.2, 174.967000),
    _Atom(72, 'Hf', 'Hafnium', 4.0, 3, 1.75, 0.3, 178.480000),
    _Atom(73, 'Ta', 'Tantalum', 5.0, 3, 1.7, 0.4, 180.947900),
    _Atom(74, 'W', 'Tungsten', 4.0, 3, 1.62, 0.3, 183.840000),
    _Atom(75, 'Re', 'Rhenium', 4.0, 3, 1.51, 0.3, 186.207000),
    _Atom(76, 'Os', 'Osmium', 4.0, 3, 1.44, 0.3, 190.230000),
    _Atom(77, 'Ir', 'Iridium', 4.0, 3, 1.41, 0.3, 192.217000),
    _Atom(78, 'Pt', 'Platinum', 4.0, 3, 1.36, 0.3, 195.078000),
    _Atom(79, 'Au', 'Gold', 1.0, 3, 1.36, 0.05, 196.966600),
    _Atom(80, 'Hg', 'Mercury', 2.0, 3, 1.32, 0.1, 200.590000),
    _Atom(81, 'Tl', 'Thallium', 3.0, 3, 1.45, 0.2, 204.383300),
    _Atom(82, 'Pb', 'Lead', 4.0, 4, 1.46, 0.3, 207.200000),
    _Atom(83, 'Bi', 'Bismuth', 3.0, 5, 1.48, 0.2, 208.980400),
    _Atom(84, 'Po', 'Polonium', 2.0, 6, 1.4, 0.2, 209.000000),
    _Atom(85, 'At', 'Astatine', 1.0, 7, 1.5, 0.1, 210.000000),
    _Atom(86, 'Rn', 'Radon', 0.5, 8, 1.5, 0.05, 222.000000),
    _Atom(87, 'Fr', 'Francium', 1.0, 1, 2.6, 0.05, 223.000000),
    _Atom(88, 'Ra', 'Radium', 2.0, 2, 2.21, 0.1, 226.000000),
    _Atom(89, 'Ac', 'Actinium', 3.0, 3, 2.15, 0.2, 227.000000),
    _Atom(90, 'Th', 'Thorium', 4.0, 3, 2.06, 0.3, 232.038100),
    _Atom(91, 'Pa', 'Protactinium', 4.0, 3, 2.0, 0.3, 231.035900),
    _Atom(92, 'U', 'Uranium', 4.0, 3, 1.96, 0.3, 238.028900),
    _Atom(93, 'Np', 'Neptunium', 4.0, 3, 1.9, 0.3, 237.000000),
    _Atom(94, 'Pu', 'Plutonium', 4.0, 3, 1.87, 0.3, 244.000000),
    _Atom(95, 'Am', 'Americium', 4.0, 3, 1.8, 0.3, 243.000000),
    _Atom(96, 'Cm', 'Curium', 4.0, 3, 1.69, 0.3, 247.000000),
    _Atom(97, 'Bk', 'Berkelium', 4.0, 3, None, 0.3, 247.000000),
    _Atom(98, 'Cf', 'Californium', 4.0, 3, None, 0.3, 251.000000),
    _Atom(99, 'Es', 'Einsteinium', 4.0, 3, None, 0.3, 252.000000),
    _Atom(100, 'Fm', 'Fermium', 4.0, 3, None, 0.3, 257.000000),
    _Atom(101, 'Md', 'Mendelevium', 4.0, 3, None, 0.3, 258.000000),
    _Atom(102, 'No', 'Nobelium', 4.0, 3, None, 0.3, 259.000000),
    _Atom(103, 'Lr', 'Lawrencium', 4.0, 3, None, 0.3, 262.000000),
    _Atom(104, 'Rf', 'Rutherfordium', 4.0, 3, None, 0.3, 261.000000),
    _Atom(105, 'Db', 'Dubnium', 2.0, 3, None, 0.1, 262.000000),
]


class Element(object):
    """
    Class for returning information about elements in one place.

    :return elements_list: all elements represented as list of tuples.
    :return z: atom number Z, e.g. 1, 5, 26, etc.
    :return short_name: short name of atom, e.g. H, He, etc.
    :return long_name: full name of the element, e.g. Iron, Oxygen, etc.
    :return valence: valence of the element.
    :return valence_electrons: valence electrons number.
    :return covalent_radius: covalent radius of the element.
    :return good_bonds: good bonds.
    :return mass: element mass.
    """

    # List with atomic number z, short name, full name, valence, valence electrons, covalent radius, good bonds:

    z = None
    short_name = None
    long_name = None
    valence = None
    valence_electrons = None
    covalent_radius = None
    good_bonds = None
    mass = None

    def __init__(self, input: Union[str, int]):
        """
        Initializes the class.

        :type input: str or int
        :param input: can be shortname, fullname of atomic number.
        """
        if isinstance(input, int):
            if max(self.all_z()) < input <= 0:
                raise ValueError
            pos = [x.z for x in _ELEMENTS_LIST].index(input)
            self.z = input
            self.short_name = _ELEMENTS_LIST[pos].shortname
            self.long_name = _ELEMENTS_LIST[pos].fullname
        elif isinstance(input, str):
            if input in self.all_short_names():
                pos = [x.shortname for x in _ELEMENTS_LIST].index(input)
                self.short_name = input
                self.z = _ELEMENTS_LIST[pos].z
                self.long_name = _ELEMENTS_LIST[pos].fullname
            elif input in self.all_long_names():
                pos = [x.fullname for x in _ELEMENTS_LIST].index(input)
                self.long_name = input
                self.short_name = _ELEMENTS_LIST[pos].shortname
                self.z = _ELEMENTS_LIST[pos].z
            else:
                raise ValueError

        self.valence = _ELEMENTS_LIST[pos].valence
        self.valence_electrons = _ELEMENTS_LIST[pos].v_electrons
        self.covalent_radius = _ELEMENTS_LIST[pos].R_covalent
        self.good_bonds = _ELEMENTS_LIST[pos].good_bonds
        self.mass = _ELEMENTS_LIST[pos].mass

    def __lt__(self, other):
        return self.z < other.z

    def __eq__(self, other):
        return self.z == other.z

    def __repr__(self):
        return self.short_name

    def __hash__(self):
        return hash(self.z)

    @staticmethod
    def all_elements() -> list:
        return [Element(x.z) for x in _ELEMENTS_LIST]

    @staticmethod
    def all_z() -> List[int]:
        return [x.z for x in _ELEMENTS_LIST]

    @staticmethod
    def all_short_names() -> List[str]:
        return [x.shortname for x in _ELEMENTS_LIST]

    @staticmethod
    def all_long_names() -> List[str]:
        return [x.fullname for x in _ELEMENTS_LIST]

    @staticmethod
    def all_valences() -> List[float]:
        return [x.valence for x in _ELEMENTS_LIST]

    @staticmethod
    def all_valence_electrons() -> List[int]:
        return [x.v_electrons for x in _ELEMENTS_LIST]

    @staticmethod
    def all_covalent_radii() -> List[float]:
        return [x.R_covalent for x in _ELEMENTS_LIST]

    @staticmethod
    def all_good_bonds() -> List[float]:
        return [x.good_bonds for x in _ELEMENTS_LIST]

    @staticmethod
    def all_masses() -> List[float]:
        return [x.mass for x in _ELEMENTS_LIST]


# -------------------------------------------------------------------------------

# if __name__ == "__main__":
#     # element = Element('Cu')
#     # element = Element(2)
#     element = Element('Hydrogen')
#     # element = Element('')
#
#     print( 'Short name:', element.short_name)
#     print( 'Z         :', element.z)
#     print( 'Long name :', element.long_name)
#     print( 'Valence   :', element.valence)
#     print( 'Val els   :', element.valence_electrons)
#     print( 'Covalent r:', element.covalent_radius)
#     print( 'Good bonds:', element.good_bonds)
#     print( 'Mass      :', element.mass)
