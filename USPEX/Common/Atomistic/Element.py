__author__ = 'mrakitin'


class Element(object):
    '''
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
    '''

    # List with atomic number z, short name, full name, valence, valence electrons, covalent radius, good bonds:
    elements_list = [
        (1, 'H', 'Hydrogen', 1.0, 1, 0.31, 0.2, 1.007940),
        (2, 'He', 'Helium', 0.5, 2, 0.28, 0.05, 4.002602),
        (3, 'Li', 'Lithium', 1.0, 1, 1.28, 0.1, 6.941000),
        (4, 'Be', 'Beryllium', 2.0, 2, 0.96, 0.2, 9.012182),
        (5, 'B', 'Boron', 3.0, 3, 0.84, 0.3, 10.811000),
        (6, 'C', 'Carbon', 4.0, 4, 0.76, 0.5, 12.010700),
        (7, 'N', 'Nitrogen', 3.0, 5, 0.71, 0.5, 14.006700),
        (8, 'O', 'Oxygen', 2.0, 6, 0.66, 0.3, 15.999400),
        (9, 'F', 'Fluorine', 1.0, 7, 0.57, 0.1, 18.998400),
        (10, 'Ne', 'Neon', 0.5, 8, 0.58, 0.05, 20.179700),
        (11, 'Na', 'Sodium', 1.0, 1, 1.66, 0.05, 22.989770),
        (12, 'Mg', 'Magnesium', 2.0, 2, 1.41, 0.1, 24.305000),
        (13, 'Al', 'Aluminium', 3.0, 3, 1.21, 0.2, 26.981540),
        (14, 'Si', 'Silicon', 4.0, 4, 1.11, 0.3, 28.085500),
        (15, 'P', 'Phosphorus', 3.0, 5, 1.07, 0.3, 30.973760),
        (16, 'S', 'Sulfur', 2.0, 6, 1.05, 0.2, 32.065000),
        (17, 'Cl', 'Chlorine', 1.0, 7, 1.02, 0.1, 35.453000),
        (18, 'Ar', 'Argon', 0.5, 8, 1.06, 0.05, 39.948000),
        (19, 'K', 'Potassium', 1.0, 1, 2.03, 0.05, 39.098300),
        (20, 'Ca', 'Calcium', 2.0, 2, 1.76, 0.1, 40.078000),
        (21, 'Sc', 'Scandium', 3.0, 3, 1.7, 0.2, 44.955910),
        (22, 'Ti', 'Titanium', 4.0, 4, 1.6, 0.3, 47.867000),
        (23, 'V', 'Vanadium', 4.0, 5, 1.53, 0.3, 50.941500),
        (24, 'Cr', 'Chromium', 3.0, 6, 1.39, 0.25, 51.996100),
        (25, 'Mn', 'Manganese', 4.0, 5, 1.39, 0.3, 54.938050),
        (26, 'Fe', 'Iron', 3.0, 3, 1.32, 0.25, 55.845000),
        (27, 'Co', 'Cobalt', 3.0, 3, 1.26, 0.25, 58.933200),
        (28, 'Ni', 'Nickel', 2.0, 3, 1.24, 0.15, 58.693200),
        (29, 'Cu', 'Copper', 2.0, 2, 1.32, 0.1, 63.546000),
        (30, 'Zn', 'Zinc', 2.0, 2, 1.22, 0.1, 65.409000),
        (31, 'Ga', 'Gallium', 3.0, 3, 1.22, 0.25, 69.723000),
        (32, 'Ge', 'Germanium', 4.0, 4, 1.2, 0.5, 72.640000),
        (33, 'As', 'Arsenic', 3.0, 5, 1.19, 0.35, 74.921600),
        (34, 'Se', 'Selenium', 2.0, 6, 1.2, 0.2, 78.960000),
        (35, 'Br', 'Bromine', 1.0, 7, 1.2, 0.1, 79.904000),
        (36, 'Kr', 'Krypton', 0.5, 8, 1.16, 0.05, 83.798000),
        (37, 'Rb', 'Rubidium', 1.0, 1, 2.2, 0.05, 86.467800),
        (38, 'Sr', 'Strontium', 2.0, 2, 1.95, 0.1, 87.620000),
        (39, 'Y', 'Yttrium', 3.0, 3, 1.9, 0.2, 88.905850),
        (40, 'Zr', 'Zirconium', 4.0, 4, 1.75, 0.3, 91.224000),
        (41, 'Nb', 'Niobium', 5.0, 5, 1.64, 0.35, 92.906380),
        (42, 'Mo', 'Molybdenum', 4.0, 6, 1.54, 0.3, 95.940000),
        (43, 'Tc', 'Technetium', 4.0, 5, 1.47, 0.3, 98.000000),
        (44, 'Ru', 'Ruthenium', 4.0, 3, 1.46, 0.3, 101.070000),
        (45, 'Rh', 'Rhodium', 4.0, 3, 1.42, 0.3, 102.905500),
        (46, 'Pd', 'Palladium', 4.0, 3, 1.39, 0.3, 106.420000),
        (47, 'Ag', 'Silver', 1.0, 2, 1.45, 0.05, 107.868200),
        (48, 'Cd', 'Cadmium', 2.0, 2, 1.44, 0.1, 112.411000),
        (49, 'In', 'Indium', 3.0, 3, 1.42, 0.2, 114.818000),
        (50, 'Sn', 'Tin', 4.0, 4, 1.39, 0.3, 118.710000),
        (51, 'Sb', 'Antimony', 3.0, 5, 1.39, 0.2, 121.760000),
        (52, 'Te', 'Tellurium', 2.0, 6, 1.38, 0.2, 127.600000),
        (53, 'I', 'Iodine', 1.0, 7, 1.39, 0.1, 126.904500),
        (54, 'Xe', 'Xenon', 0.5, 8, 1.4, 0.05, 131.293000),
        (55, 'Cs', 'Caesium', 1.0, 1, 2.44, 0.05, 132.905500),
        (56, 'Ba', 'Barium', 2.0, 2, 2.15, 0.1, 137.327000),
        (57, 'La', 'Lanthanum', 3.0, 3, 2.07, 0.2, 138.905500),
        (58, 'Ce', 'Cerium', 4.0, 3, 2.04, 0.3, 140.116000),
        (59, 'Pr', 'Praseodymium', 3.0, 3, 2.03, 0.2, 140.907700),
        (60, 'Nd', 'Neodymium', 3.0, 3, 2.01, 0.2, 144.240000),
        (61, 'Pm', 'Promethium', 3.0, 3, 1.99, 0.2, 145.000000),
        (62, 'Sm', 'Samarium', 3.0, 3, 1.98, 0.2, 150.360000),
        (63, 'Eu', 'Europium', 3.0, 3, 1.98, 0.2, 151.964000),
        (64, 'Gd', 'Gadolinium', 3.0, 3, 1.96, 0.2, 157.250000),
        (65, 'Tb', 'Terbium', 3.0, 3, 1.94, 0.2, 158.925300),
        (66, 'Dy', 'Dysprosium', 3.0, 3, 1.92, 0.2, 162.500000),
        (67, 'Ho', 'Holmium', 3.0, 3, 1.92, 0.2, 164.930300),
        (68, 'Er', 'Erbium', 3.0, 3, 1.89, 0.2, 167.259000),
        (69, 'Tm', 'Thulium', 3.0, 3, 1.9, 0.2, 168.934200),
        (70, 'Yb', 'Ytterbium', 3.0, 3, 1.87, 0.2, 173.040000),
        (71, 'Lu', 'Lutetium', 3.0, 3, 1.87, 0.2, 174.967000),
        (72, 'Hf', 'Hafnium', 4.0, 3, 1.75, 0.3, 178.480000),
        (73, 'Ta', 'Tantalum', 5.0, 3, 1.7, 0.4, 180.947900),
        (74, 'W', 'Tungsten', 4.0, 3, 1.62, 0.3, 183.840000),
        (75, 'Re', 'Rhenium', 4.0, 3, 1.51, 0.3, 186.207000),
        (76, 'Os', 'Osmium', 4.0, 3, 1.44, 0.3, 190.230000),
        (77, 'Ir', 'Iridium', 4.0, 3, 1.41, 0.3, 192.217000),
        (78, 'Pt', 'Platinum', 4.0, 3, 1.36, 0.3, 195.078000),
        (79, 'Au', 'Gold', 1.0, 3, 1.36, 0.05, 196.966600),
        (80, 'Hg', 'Mercury', 2.0, 3, 1.32, 0.1, 200.590000),
        (81, 'Tl', 'Thallium', 3.0, 3, 1.45, 0.2, 204.383300),
        (82, 'Pb', 'Lead', 4.0, 4, 1.46, 0.3, 207.200000),
        (83, 'Bi', 'Bismuth', 3.0, 5, 1.48, 0.2, 208.980400),
        (84, 'Po', 'Polonium', 2.0, 6, 1.4, 0.2, 209.000000),
        (85, 'At', 'Astatine', 1.0, 7, 1.5, 0.1, 210.000000),
        (86, 'Rn', 'Radon', 0.5, 8, 1.5, 0.05, 222.000000),
        (87, 'Fr', 'Francium', 1.0, 1, 2.6, 0.05, 223.000000),
        (88, 'Ra', 'Radium', 2.0, 2, 2.21, 0.1, 226.000000),
        (89, 'Ac', 'Actinium', 3.0, 3, 2.15, 0.2, 227.000000),
        (90, 'Th', 'Thorium', 4.0, 3, 2.06, 0.3, 232.038100),
        (91, 'Pa', 'Protactinium', 4.0, 3, 2.0, 0.3, 231.035900),
        (92, 'U', 'Uranium', 4.0, 3, 1.96, 0.3, 238.028900),
        (93, 'Np', 'Neptunium', 4.0, 3, 1.9, 0.3, 237.000000),
        (94, 'Pu', 'Plutonium', 4.0, 3, 1.87, 0.3, 244.000000),
        (95, 'Am', 'Americium', 4.0, 3, 1.8, 0.3, 243.000000),
        (96, 'Cm', 'Curium', 4.0, 3, 1.69, 0.3, 247.000000),
        (97, 'Bk', 'Berkelium', 4.0, 3, None, 0.3, 247.000000),
        (98, 'Cf', 'Californium', 4.0, 3, None, 0.3, 251.000000),
        (99, 'Es', 'Einsteinium', 4.0, 3, None, 0.3, 252.000000),
        (100, 'Fm', 'Fermium', 4.0, 3, None, 0.3, 257.000000),
        (101, 'Md', 'Mendelevium', 4.0, 3, None, 0.3, 258.000000),
        (102, 'No', 'Nobelium', 4.0, 3, None, 0.3, 259.000000),
        (103, 'Lr', 'Lawrencium', 4.0, 3, None, 0.3, 262.000000),
        (104, 'Rf', 'Rutherfordium', 4.0, 3, None, 0.3, 261.000000),
        (105, 'Db', 'Dubnium', 2.0, 3, None, 0.1, 262.000000),
    ]

    z = None
    short_name = None
    long_name = None
    valence = None
    valence_electrons = None
    covalent_radius = None
    good_bonds = None
    mass = None

    def __init__(self, input_value):
        self.input = input_value

        pos = None

        try:
            int(self.input)
            self.z = self.input

            for i, el in enumerate(self.elements_list):
                if el[0] == self.z:
                    pos = i
                    self.short_name = el[1]
                    self.long_name = el[2]
                    break
        except ValueError:
            self.short_name = self.input
            for i, el in enumerate(self.elements_list):
                if el[1] == self.short_name:
                    pos = i
                    self.z = el[0]
                    self.long_name = el[2]
                    break

            if not self.z:
                self.short_name = None
                self.long_name = self.input
                for i, el in enumerate(self.elements_list):
                    if el[2] == self.long_name:
                        pos = i
                        self.z = el[0]
                        self.short_name = el[1]
                        break
                if not self.z:
                    self.long_name = None

        if pos is not None:
            self.valence = self.elements_list[pos][3]
            self.valence_electrons = self.elements_list[pos][4]
            self.covalent_radius = self.elements_list[pos][5]
            self.good_bonds = self.elements_list[pos][6]
            self.mass = self.elements_list[pos][7]

    def get_all(self, pos):
        els = []
        for el in self.elements_list:
            els.append(el[pos])
        return els

    def all_z(self):
        return self.get_all(0)

    def all_short_names(self):
        return self.get_all(1)

    def all_long_names(self):
        return self.get_all(2)

    def all_valences(self):
        return self.get_all(3)

    def all_valence_electrons(self):
        return self.get_all(4)

    def all_covalent_radii(self):
        return self.get_all(5)

    def all_good_bonds(self):
        return self.get_all(6)

    def all_masses(self):
        return self.get_all(7)


# -------------------------------------------------------------------------------

if __name__ == "__main__":
    # element = Element('Cu')
    # element = Element(2)
    element = Element('Hydrogen')
    # element = Element('')

    print( 'Short name:', element.short_name)
    print( 'Z         :', element.z)
    print( 'Long name :', element.long_name)
    print( 'Valence   :', element.valence)
    print( 'Val els   :', element.valence_electrons)
    print( 'Covalent r:', element.covalent_radius)
    print( 'Good bonds:', element.good_bonds)
    print( 'Mass      :', element.mass)

    print( element.all_z())
    print( element.all_short_names())
    print( element.all_long_names())
    print( element.all_valences())
    print( element.all_valence_electrons())
    print( element.all_covalent_radii())
    print( element.all_good_bonds())
    print( element.all_masses())
