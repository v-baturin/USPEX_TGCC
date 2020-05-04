"""
USPEX.Common.Atomistic.Fingerprints.Fingerprints
================================================

Class denoting a fingerprint

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import numpy as np
from .make_matrices import make_matrices
from .fingerprint import fingerprint
from .quasientropy import quasientropy
from .structure_order import structure_order


RMAX_DEFAULT = 10.0
SIGMA_DEFAULT = 0.03
DELTA_DEFAULT = 0.08

TOLERANCE_DEFAULT = 0.008


class Fingerprints:
    """
    :ivar fingerprint:
    :ivar atom_fingerprint:
    :ivar quasientropy:
    :ivar order:
    :ivar a_order:
    :ivar s_order:
    :ivar weight:
    """

    def __init__(self, structure, Rmax=RMAX_DEFAULT, sigma=SIGMA_DEFAULT, delta=DELTA_DEFAULT, **kwargs):
        """

        :type structure: :class:`~USPEX.Common.Atomistic.AtomicStructure.AtomicStructure` or descendant
        :param structure: system for which we want to calculate fingerprint.
        :type Rmax: float
        :param Rmax: threshold distance between i-th anf j-th atom.
        :type sigma: float
        :param sigma: smearing parameter.
        :type delta: float
        :param delta: bin width.
        :type kwargs: dict
        :param kwargs: additional arguments.
        """
        self.structure = structure
        self.fingerprint = None
        self.atom_fingerprint = None
        self.quasientropy = None
        self.order = None
        self.a_order = None
        self.s_order = None
        self.volume = None
        self.dist_matrix = None
        self.weight = None

        if type(Rmax) == float or type(Rmax) == int:
            self.Rmax = float(Rmax)
        else:
            self.Rmax = RMAX_DEFAULT

        if type(sigma) == float or type(sigma) == int:
            self.sigma = float(sigma)
        else:
            self.sigma = SIGMA_DEFAULT

        if type(delta) == float or type(delta) == int:
            self.delta = float(delta)
        else:
            self.delta = DELTA_DEFAULT

        self.outfile = None

        self.volume = structure.get_volume()
        self.dist_matrix = make_matrices(self.structure, Rmax=self.Rmax)
        order, self.fingerprint, self.atom_fingerprint = fingerprint(self.volume, self.dist_matrix,
                                                                     self.structure, Rmax=self.Rmax,
                                                                     sigma=self.sigma,
                                                                     delta=self.delta)
        self.order = np.zeros(order.shape, dtype=float)
        inds = np.argsort(structure.get_chemical_symbols())
        for i, ord in enumerate(order):
            self.order[inds[i]] = ord

        if np.any(np.isfinite(self.order)):
            self.a_order = np.mean(self.order[np.isfinite(self.order)])
        else:
            self.a_order = np.nan

        self.quasientropy = quasientropy(structure, self.atom_fingerprint)
        self.weight = fp_weight(structure)
        self.s_order = structure_order(self.fingerprint, self.volume, structure, self.weight, self.delta)
        self.system = structure.get_chemical_formula()

        self.info = 'System: {:s}    Quasientropy: {:.4f}    A-order: {:.4f}    S-order: {:.4f}'.format(
            self.system, self.quasientropy, self.a_order, self.s_order)

        self.system_plain = self.system.replace('$_{', '').replace('}$', '')
        self.info_plain = self.info.replace('$_{', '').replace('}$', '')

    def plot(self, file_name=None, include_infile=None):
        from matplotlib import pyplot as plt
        # Outfile for the resulted figure:
        self.outfile = 'fingerprints.png'
        if self.system_plain:
            self.outfile = self.structure.get_chemical_formula() + '_' + self.system_plain + '_' + self.outfile
        if file_name:
            self.outfile = file_name

        if not include_infile:
            self.info = 'System: %s     Quasientropy: %.4f    A-order: %.4f    S-order: %.4f' % (
                self.system, self.quasientropy, self.a_order, self.s_order)

        fig = plt.figure(figsize=(16, 10))

        elements_num = np.unique(self.structure.get_chemical_symbols(), return_counts=True)[1].shape[0]
        m = 1  # counter of unique combinations of elements
        total_axes = elements_num * (elements_num + 1) / 2  # total number of unique plots
        x = np.linspace(0, self.Rmax, self.fingerprint.shape[1])  # to plot x-axis from 0 to Rmax

        for i in range(elements_num):
            for j in range(elements_num):
                k = i * elements_num + j  # index in 1-d array
                if i <= j:
                    ax = fig.add_subplot(total_axes, 1, m)
                    if k == 0:
                        ax.set_title(self.info)
                    ax.set_ylabel(str(np.unique(self.structure.get_chemical_symbols(), return_counts=True)[1][i]) +
                                  '-' + str(np.unique(self.structure.get_chemical_symbols(), return_counts=True)[1][j]))

                    if self.Rmax <= 25.0:
                        ticks_step = 1.0
                    else:
                        ticks_step = self.Rmax / 10.0

                    ax.set_xticks(np.arange(x[0], x[-1] + 1, ticks_step))
                    ax.grid()
                    ax.plot(x, self.fingerprint[k, :])
                    m += 1

        plt.xlabel(r"R, ${\AA}$")
        # plt.show()
        fig.tight_layout()
        fig.savefig(self.outfile, dpi=150)
        pass


if __name__ == '__main__':
    from ase.io import read
    system = read('POSCAR')
    f = Fingerprints(system)
    print(f.info_plain)
    f.plot()
