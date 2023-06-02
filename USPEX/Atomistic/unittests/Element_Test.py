"""
USPEX.Common.Atomistic.unittests.Element_Test
=============================================

Class for Element testing

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import unittest

from ..Element import Element


class Element_Test(unittest.TestCase):

    def test_static_methods(self):
        assert len(Element.all_z()) == 105
        assert len(Element.all_short_names()) == 105
        assert len(Element.all_long_names()) == 105
        assert len(Element.all_valences()) == 105
        assert len(Element.all_valence_electrons()) == 105
        assert len(Element.all_covalent_radii()) == 105
        assert len(Element.all_good_bonds()) == 105
        assert len(Element.all_masses()) == 105

    def test_element_by_shortname(self):
        name = 'Si'
        element = Element(name)
        self.assertEqual(element.z, 14)
        self.assertEqual(element.long_name, 'Silicon')

    def test_element_by_fullname(self):
        name = 'Silicon'
        element = Element(name)
        self.assertEqual(element.z, 14)
        self.assertEqual(element.short_name, 'Si')

    def test_element_by_z(self):
        z = 14
        element = Element(z)
        self.assertEqual(element.z, 14)
        self.assertEqual(element.short_name, 'Si')
        self.assertEqual(element.long_name, 'Silicon')

    def test_bad_name(self):
        name = 'phd'
        self.assertRaises(ValueError, Element, name)

    def test_bad_z(self):
        Z = 100000
        self.assertRaises(ValueError, Element, Z)
