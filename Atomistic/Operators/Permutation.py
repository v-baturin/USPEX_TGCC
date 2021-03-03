import numpy as np
from itertools import combinations
from copy import copy

class Permutation:

    def __init__(self, utilities, howManySwaps = 5):
        self.specificSwaps = []
        self.howManySwaps = howManySwaps

    def __call__(self, system, *args, **kwargs):
        molecules = system['molecules']
        indices = list(range(len(molecules)))
        symbols = [molecule.symbol for molecule in molecules]
        excluded = [{s,s} for s in np.unique(symbols)]

        swaps = [{i1,i2} for (i1,i2) in combinations(indices, 2) if {symbols[i1], symbols[i2]} not in excluded]

        permutations = []
        for numberOfSwaps in range(1, self.howManySwaps + 1):
            for permutation in combinations(swaps, numberOfSwaps):
                # ensure that every molecule swapped only once
                if len(set.union(*permutation)) == numberOfSwaps*2:
                    permutations.append(permutation)


        for permutation in np.random.permutation(permutations):
            attempted_structure = copy(molecules)

            # It's boring to permute molecules just one time. Let's do it from 1 to 'howManySwaps' times! Default howManySwaps is 5.
            for s1, s2 in permutation:
                # Get first permutation from the total list of permutations; then permute
                # Generate translation vector for molecules (geometric center of molecule 1 minus geometric center of molecule 2,
                # it's regular linear algebra).
                translation_1to2 = attempted_structure[s1].get_center_of_mass(scaled=True) - \
                                   attempted_structure[s2].get_center_of_mass(scaled=True)
                # Translate molecule 1 to achieve coincidence of it's geometric center with geometric center of molecule 2
                attempted_structure[s1].translate_scaled(translation_1to2)
                # ...and the other way around.
                attempted_structure[s2].translate_scaled(-translation_1to2)
                atLeastOnePermutation = True
