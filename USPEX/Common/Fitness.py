from typing import List, Tuple

def fitnessHash(fitness : List[Tuple[str, str]]):
    return hash('_'.join(f'{attr}_{direction}' for (attr, direction) in sorted(fitness, key=lambda entry: entry[0])))