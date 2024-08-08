from typing import Callable, Tuple, Optional
from inspect import signature
import numpy as np
from numpy.typing import NDArray

class GO_testing_function:

    def __init__(self, fn: Callable,
                 global_optima: Tuple[Tuple[Tuple[float, ...], float], ...],
                 domain_borders: Optional[Tuple[Tuple[float, float], ...]] = None):
        """
        :param fn: callable object with arbitrary number N of positional NDArray arguments
        :param global_optima: (((x1_min1, x2_min1, ... xN_min1), val_1),
                               ((x1_min2, x2_min2, ... xN_min2), val_2),...
                               ((x1_minM, x2_minM, ... xN_minM), val_M)) -- M known global minima (val1 = ... = valM)
                               fn(*global_optima[k][0]) = val_k
                               xi_mink - float
        :param domain_borders: ((a1, b1), ... ,(aN, bN)) -- standard domain of search.
        """
        self.fn = fn
        self.global_optima = global_optima
        self.domain_borders = domain_borders

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def affine_transform(self, k: Tuple[float, ...] = None, b: Tuple[float, ...] = None):
        n_par = len(signature(self.fn).parameters)
        k = (1.,) * n_par if k is None else k
        b = (0.,) * n_par if b is None else b

        def affine(args: Tuple[Tuple[float, ...] | float, ...],
                   coeff: Tuple[float, ...] = k,
                   shift: Tuple[float, ...] = b) -> Tuple[Tuple | float | NDArray, ...]:
            res = []
            for arg_i, k_i, b_i in zip(args, coeff, shift):
                res_i = tuple(k_i * np.array(arg_i) + b_i) if isinstance(arg_i, tuple) else k_i * np.array(arg_i) + b_i
                res.append(res_i)
            return tuple(res)

        def rev_affine(*args, coeff: Tuple[float, ...] = k, shift: Tuple[float, ...] = b) -> Tuple[Tuple | float | NDArray, ...]:
            return affine(args,
                          coeff=tuple(1/k_i for k_i in coeff),
                          shift=tuple(- b_i/k_i for k_i, b_i in zip(coeff, shift)))
        """
        f(x, y) -> f(t, s) = f(k1 * x + b1, k2 * y+b2)
        :param k: coefficient(s)
        :param b: displacement(s)
        :return:
        """
        original_fn = self.fn

        def transformed_fn(*args):
            return original_fn(*affine(args))

        self.fn = transformed_fn

        # Update global optima:
        self.global_optima = tuple((rev_affine(*i_go[0]), i_go[1]) for i_go in self.global_optima)

        # Update domain borders
        self.domain_borders = rev_affine(*self.domain_borders)

    def transform_to_match_new_borders(self, new_borders: Tuple[Tuple[float, float], ...]):
        k_s = []
        b_s = []
        for old_ab_i, new_ab_i in zip(self.domain_borders, new_borders):
            k = (old_ab_i[1] - old_ab_i[0]) / (new_ab_i[1] - new_ab_i[0])
            k_s.append(k)
            b_s.append(old_ab_i[0] - k * new_ab_i[0])
        self.affine_transform(tuple(k_s), tuple(b_s))

    def wrap_by(self, wrapping_fn: Callable):
        original_fn = self.fn
        transformed_fn = lambda *args: wrapping_fn(original_fn(*args))
        self.fn = transformed_fn
        self.global_optima = tuple((go_arg, self.fn(*go_arg)) for go_arg, _ in self.global_optima)




function_lib = {'Himmelblau_xy': {'fn': lambda x, y: (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2,
                                  'global_optima': tuple(((xi, yi), 0.)
                                                         for xi, yi in [[3., 2.],
                                                                        [-2.805118, 3.131312],
                                                                        [-3.779310, -3.283186],
                                                                        [3.584428, -1.848126]]),
                                  'domain_borders': ((-5., 5), (-5., 5))
                                  },
                'quadratic_xy': {'fn': lambda x, y: x ** 2 + y ** 2,
                                 'global_optima': (((0., 0.), 0.0),),
                                 'domain_borders': ((-2., 2), (-2., 2))},
                }

if __name__ == '__main__':
    # hb_xy = GO_testing_function(**function_lib['Himmelblau_xy'])
    # print(hb_xy(3., 2))
    # hb_xy.affine_transform()
    # print(hb_xy(3, 2))
    # hb_vec = GO_testing_function(**function_lib['Himmelblau_vec'])
    # print(hb_vec(np.array([3., 2.])))
    # hb_vec.affine_transform()
    # print(hb_vec(np.array([3., 2.])))
    # hb_vec.wrap_by(lambda x: x - 100)
    # print(hb_vec(np.array([3., 2.])))
    q_vec = GO_testing_function(**function_lib['quadratic_vec'])
    # print(q_vec(*q_vec.global_optima[0][0]))
    q_xy = GO_testing_function(**function_lib['quadratic_xy'])
    print(q_xy(*q_xy.global_optima[0][0]))
    q_xy.transform_to_match_new_borders(((5., 7.), (-10., -8.)))
    print(q_xy.global_optima[0][0])
