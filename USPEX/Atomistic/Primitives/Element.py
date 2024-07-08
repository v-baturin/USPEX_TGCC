"""
USPEX.Atomistic.Element
==============================

Class for Element

.. codeauthor:: Maxim Rakitin
.. codeauthor:: Artem Samtsevich <samtsevichartem@gmail.com>
"""

from typing import List, Union, Optional

from ...Semantics.Atomistic.Primitives.Element import Element as ElementSemantics

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
                 R_covalent: Optional[float], R_vdW: Optional[float], good_bonds: float, mass: float):
        self.z = z
        self.shortname = shortname
        self.fullname = fullname
        self.valence = valence
        self.v_electrons = v_electrons
        self.R_covalent = R_covalent
        self.R_vdW = R_vdW
        self.good_bonds = good_bonds
        self.mass = mass


_ELEMENTS_LIST = [
    _Atom(1, 'H', 'Hydrogen', 1.00, 1, 0.31, 1.20, 0.20, 1.007940),
    _Atom(2, 'He', 'Helium', 0.50, 2, 0.28, 1.40, 0.05, 4.002602),
    _Atom(3, 'Li', 'Lithium', 1.00, 1, 1.28, 1.82, 0.10, 6.941000),
    _Atom(4, 'Be', 'Beryllium', 2.00, 2, 0.96, 1.53, 0.20, 9.012182),
    _Atom(5, 'B', 'Boron', 3.00, 3, 0.84, 1.92, 0.30, 10.811000),
    _Atom(6, 'C', 'Carbon', 4.00, 4, 0.76, 1.70, 0.50, 12.010700),
    _Atom(7, 'N', 'Nitrogen', 3.00, 5, 0.71, 1.55, 0.50, 14.006700),
    _Atom(8, 'O', 'Oxygen', 2.00, 6, 0.66, 1.52, 0.30, 15.999400),
    _Atom(9, 'F', 'Fluorine', 1.00, 7, 0.57, 1.47, 0.10, 18.998400),
    _Atom(10, 'Ne', 'Neon', 0.50, 8, 0.58, 1.54, 0.05, 20.179700),
    _Atom(11, 'Na', 'Sodium', 1.00, 1, 1.66, 2.27, 0.05, 22.989770),
    _Atom(12, 'Mg', 'Magnesium', 2.00, 2, 1.41, 1.73, 0.10, 24.305000),
    _Atom(13, 'Al', 'Aluminium', 3.00, 3, 1.21, 1.84, 0.20, 26.981540),
    _Atom(14, 'Si', 'Silicon', 4.00, 4, 1.11, 2.10, 0.30, 28.085500),
    _Atom(15, 'P', 'Phosphorus', 3.00, 5, 1.07, 1.80, 0.30, 30.973760),
    _Atom(16, 'S', 'Sulfur', 2.00, 6, 1.05, 1.80, 0.20, 32.065000),
    _Atom(17, 'Cl', 'Chlorine', 1.00, 7, 1.02, 1.75, 0.10, 35.453000),
    _Atom(18, 'Ar', 'Argon', 0.50, 8, 1.06, 1.88, 0.05, 39.948000),
    _Atom(19, 'K', 'Potassium', 1.00, 1, 2.03, 2.75, 0.05, 39.098300),
    _Atom(20, 'Ca', 'Calcium', 2.00, 2, 1.76, 2.31, 0.10, 40.078000),
    _Atom(21, 'Sc', 'Scandium', 3.00, 3, 1.70, 2.11, 0.20, 44.955910),
    _Atom(22, 'Ti', 'Titanium', 4.00, 4, 1.60, 2.52, 0.30, 47.867000),
    _Atom(23, 'V', 'Vanadium', 4.00, 5, 1.53, 2.05, 0.30, 50.941500),
    _Atom(24, 'Cr', 'Chromium', 3.00, 6, 1.39, 2.05, 0.25, 51.996100),
    _Atom(25, 'Mn', 'Manganese', 4.00, 5, 1.39, 2.05, 0.30, 54.938050),
    _Atom(26, 'Fe', 'Iron', 3.00, 3, 1.32, 2.05, 0.25, 55.845000),
    _Atom(27, 'Co', 'Cobalt', 3.00, 3, 1.26, 2.00, 0.25, 58.933200),
    _Atom(28, 'Ni', 'Nickel', 2.00, 3, 1.24, 2.00, 0.15, 58.693200),
    _Atom(29, 'Cu', 'Copper', 2.00, 2, 1.32, 2.00, 0.10, 63.546000),
    _Atom(30, 'Zn', 'Zinc', 2.00, 2, 1.22, 2.10, 0.10, 65.409000),
    _Atom(31, 'Ga', 'Gallium', 3.00, 3, 1.22, 2.10, 0.25, 69.723000),
    _Atom(32, 'Ge', 'Germanium', 4.00, 4, 1.20, 2.10, 0.50, 72.640000),
    _Atom(33, 'As', 'Arsenic', 3.00, 5, 1.19, 2.05, 0.35, 74.921600),
    _Atom(34, 'Se', 'Selenium', 2.00, 6, 1.20, 1.90, 0.20, 78.960000),
    _Atom(35, 'Br', 'Bromine', 1.00, 7, 1.20, 1.85, 0.10, 79.904000),
    _Atom(36, 'Kr', 'Krypton', 0.50, 8, 1.16, 2.02, 0.05, 83.798000),
    _Atom(37, 'Rb', 'Rubidium', 1.00, 1, 2.20, 3.03, 0.05, 86.467800),
    _Atom(38, 'Sr', 'Strontium', 2.00, 2, 1.95, 2.49, 0.10, 87.620000),
    _Atom(39, 'Y', 'Yttrium', 3.00, 3, 1.90, 2.40, 0.20, 88.905850),
    _Atom(40, 'Zr', 'Zirconium', 4.00, 4, 1.75, 2.30, 0.30, 91.224000),
    _Atom(41, 'Nb', 'Niobium', 5.00, 5, 1.64, 2.15, 0.35, 92.906380),
    _Atom(42, 'Mo', 'Molybdenum', 4.00, 6, 1.54, 2.10, 0.30, 95.940000),
    _Atom(43, 'Tc', 'Technetium', 4.00, 5, 1.47, 2.05, 0.30, 98.000000),
    _Atom(44, 'Ru', 'Ruthenium', 4.00, 3, 1.46, 2.05, 0.30, 101.070000),
    _Atom(45, 'Rh', 'Rhodium', 4.00, 3, 1.42, 2.00, 0.30, 102.905500),
    _Atom(46, 'Pd', 'Palladium', 4.00, 3, 1.39, 2.05, 0.30, 106.420000),
    _Atom(47, 'Ag', 'Silver', 1.00, 2, 1.45, 2.10, 0.05, 107.868200),
    _Atom(48, 'Cd', 'Cadmium', 2.00, 2, 1.44, 2.20, 0.10, 112.411000),
    _Atom(49, 'In', 'Indium', 3.00, 3, 1.42, 2.20, 0.20, 114.818000),
    _Atom(50, 'Sn', 'Tin', 4.00, 4, 1.39, 2.25, 0.30, 118.710000),
    _Atom(51, 'Sb', 'Antimony', 3.00, 5, 1.39, 2.20, 0.20, 121.760000),
    _Atom(52, 'Te', 'Tellurium', 2.00, 6, 1.38, 2.10, 0.20, 127.600000),
    _Atom(53, 'I', 'Iodine', 1.00, 7, 1.39, 2.10, 0.10, 126.904500),
    _Atom(54, 'Xe', 'Xenon', 0.50, 8, 1.40, 1.31, 0.05, 131.293000),
    _Atom(55, 'Cs', 'Caesium', 1.00, 1, 2.44, 3.00, 0.05, 132.905500),
    _Atom(56, 'Ba', 'Barium', 2.00, 2, 2.15, 2.70, 0.10, 137.327000),
    _Atom(57, 'La', 'Lanthanum', 3.00, 3, 2.07, 2.50, 0.20, 138.905500),
    _Atom(58, 'Ce', 'Cerium', 4.00, 3, 2.04, 2.21, 0.30, 140.116000),
    _Atom(59, 'Pr', 'Praseodymium', 3.00, 3, 2.03, 2.18, 0.20, 140.907700),
    _Atom(60, 'Nd', 'Neodymium', 3.00, 3, 2.01, 2.71, 0.20, 144.240000),
    _Atom(61, 'Pm', 'Promethium', 3.00, 3, 1.99, 2.73, 0.20, 145.000000),
    _Atom(62, 'Sm', 'Samarium', 3.00, 3, 1.98, 2.70, 0.20, 150.360000),
    _Atom(63, 'Eu', 'Europium', 3.00, 3, 1.98, 2.70, 0.20, 151.964000),
    _Atom(64, 'Gd', 'Gadolinium', 3.00, 3, 1.96, 2.22, 0.20, 157.250000),
    _Atom(65, 'Tb', 'Terbium', 3.00, 3, 1.94, 2.67, 0.20, 158.925300),
    _Atom(66, 'Dy', 'Dysprosium', 3.00, 3, 1.92, 2.68, 0.20, 162.500000),
    _Atom(67, 'Ho', 'Holmium', 3.00, 3, 1.92, 2.66, 0.20, 164.930300),
    _Atom(68, 'Er', 'Erbium', 3.00, 3, 1.89, 2.66, 0.20, 167.259000),
    _Atom(69, 'Tm', 'Thulium', 3.00, 3, 1.90, 2.66, 0.20, 168.934200),
    _Atom(70, 'Yb', 'Ytterbium', 3.00, 3, 1.87, 2.66, 0.20, 173.040000),
    _Atom(71, 'Lu', 'Lutetium', 3.00, 3, 1.87, 2.21, 0.20, 174.967000),
    _Atom(72, 'Hf', 'Hafnium', 4.00, 3, 1.75, 2.25, 0.30, 178.480000),
    _Atom(73, 'Ta', 'Tantalum', 5.00, 3, 1.70, 2.20, 0.40, 180.947900),
    _Atom(74, 'W', 'Tungsten', 4.00, 3, 1.62, 2.10, 0.30, 183.840000),
    _Atom(75, 'Re', 'Rhenium', 4.00, 3, 1.51, 2.05, 0.30, 186.207000),
    _Atom(76, 'Os', 'Osmium', 4.00, 3, 1.44, 2.00, 0.30, 190.230000),
    _Atom(77, 'Ir', 'Iridium', 4.00, 3, 1.41, 2.00, 0.30, 192.217000),
    _Atom(78, 'Pt', 'Platinum', 4.00, 3, 1.36, 2.05, 0.30, 195.078000),
    _Atom(79, 'Au', 'Gold', 1.00, 3, 1.36, 2.10, 0.05, 196.966600),
    _Atom(80, 'Hg', 'Mercury', 2.00, 3, 1.32, 2.05, 0.10, 200.590000),
    _Atom(81, 'Tl', 'Thallium', 3.00, 3, 1.45, 2.20, 0.20, 204.383300),
    _Atom(82, 'Pb', 'Lead', 4.00, 4, 1.46, 2.30, 0.30, 207.200000),
    _Atom(83, 'Bi', 'Bismuth', 3.00, 5, 1.48, 2.30, 0.20, 208.980400),
    _Atom(84, 'Po', 'Polonium', 2.00, 6, 1.40, 1.97, 0.20, 209.000000),
    _Atom(85, 'At', 'Astatine', 1.00, 7, 1.50, 2.02, 0.10, 210.000000),
    _Atom(86, 'Rn', 'Radon', 0.50, 8, 1.50, 2.20, 0.05, 222.000000),
    _Atom(87, 'Fr', 'Francium', 1.00, 1, 2.60, 3.48, 0.05, 223.000000),
    _Atom(88, 'Ra', 'Radium', 2.00, 2, 2.21, 2.83, 0.10, 226.000000),
    _Atom(89, 'Ac', 'Actinium', 3.00, 3, 2.15, 3.05, 0.20, 227.000000),
    _Atom(90, 'Th', 'Thorium', 4.00, 3, 2.06, 2.40, 0.30, 232.038100),
    _Atom(91, 'Pa', 'Protactinium', 4.00, 3, 2.00, 2.53, 0.30, 231.035900),
    _Atom(92, 'U', 'Uranium', 4.00, 3, 1.96, 2.80, 0.30, 238.028900),
    _Atom(93, 'Np', 'Neptunium', 4.00, 3, 1.90, 2.45, 0.30, 237.000000),
    _Atom(94, 'Pu', 'Plutonium', 4.00, 3, 1.87, 2.49, 0.30, 244.000000),
    _Atom(95, 'Am', 'Americium', 4.00, 3, 1.80, 2.63, 0.30, 243.000000),
    _Atom(96, 'Cm', 'Curium', 4.00, 3, 1.69, 2.64, 0.30, 247.000000),
    _Atom(97, 'Bk', 'Berkelium', 4.00, 3, None, None, 0.30, 247.000000),
    _Atom(98, 'Cf', 'Californium', 4.00, 3, None, None, 0.30, 251.000000),
    _Atom(99, 'Es', 'Einsteinium', 4.00, 3, None, None, 0.30, 252.000000),
    _Atom(100, 'Fm', 'Fermium', 4.00, 3, None, None, 0.30, 257.000000),
    _Atom(101, 'Md', 'Mendelevium', 4.00, 3, None, None, 0.30, 258.000000),
    _Atom(102, 'No', 'Nobelium', 4.00, 3, None, None, 0.30, 259.000000),
    _Atom(103, 'Lr', 'Lawrencium', 4.00, 3, None, None, 0.30, 262.000000),
    _Atom(104, 'Rf', 'Rutherfordium', 4.00, 3, None, None, 0.30, 261.000000),
    _Atom(105, 'Db', 'Dubnium', 2.00, 3, None, None, 0.10, 262.000000)
]


class Element(ElementSemantics):
    """
    Class for returning information about elements in one place.

    :ivar z: atom number Z, e.g. 1, 5, 26, etc.
    :ivar short_name: short name of atom, e.g. H, He, etc.
    :ivar long_name: full name of the element, e.g. Iron, Oxygen, etc.
    :ivar valence: valence of the element.
    :ivar valence_electrons: valence electrons number.
    :ivar covalent_radius: covalent radius of the element.
    :ivar good_bonds: good bonds.
    :ivar mass: element mass.
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

    def __init__(self, input: Union[str, int], charge=None, **kwargs):
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
        elif isinstance(input, Element):
            self.short_name = input.short_name
            self.z = input.z
            self.long_name = input.long_name
            pos = [x.z for x in _ELEMENTS_LIST].index(input.z)

        self.valence = _ELEMENTS_LIST[pos].valence
        self.valence_electrons = _ELEMENTS_LIST[pos].v_electrons
        self.covalent_radius = _ELEMENTS_LIST[pos].R_covalent
        self.vanderWaals_radius = _ELEMENTS_LIST[pos].R_vdW
        self.good_bonds = _ELEMENTS_LIST[pos].good_bonds
        self.mass = _ELEMENTS_LIST[pos].mass
        self.charge = charge
        self.extra = kwargs


    def __lt__(self, other):
        return self.z < other.z

    def __eq__(self, other):
        return self.z == other.z

    def __repr__(self):
        return self.short_name

    def __hash__(self):
        return hash(self.z)

    def extendedRepresentation(self):
        rep = f'name: {self.short_name}'
        if self.charge is not None:
            rep += f', charge: {self.charge}'
        for key, value in self.extra.items():
            rep += f', {key}: {value}'
        return f'{{{rep}}}'

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
