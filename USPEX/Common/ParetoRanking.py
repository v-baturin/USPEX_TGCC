import numpy as np

def paretoRanking(fitnesses):
    fitnesses = np.asarray(fitnesses)
    N = fitnesses.shape[1] # number of fitnesses
    fronts = []
    unranked = np.arange(len(fitnesses))
    while len(unranked) > 0:
        A = np.all(fitnesses[unranked].reshape((len(unranked), 1, N)) >=
                   fitnesses[unranked].reshape((1, len(unranked), N)), axis=2)
        B = np.logical_not(np.all(fitnesses[unranked].reshape((len(unranked), 1, N)) ==
                                  fitnesses[unranked].reshape((1, len(unranked), N)), axis=2))
        front = np.flatnonzero(np.logical_not(np.any(np.logical_and(A, B), axis=1)))
        fronts.append(unranked[front])
        unranked = np.asarray(list(set(unranked) - set(unranked[front])))
    return fronts
