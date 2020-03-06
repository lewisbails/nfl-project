import numpy as np
import pandas as pd
from scipy.stats import chi2, pointbiserialr, pearsonr
import itertools


def tetrachoric(c1, c2):
    c11 = c1 == 1
    c10 = c1 == 0
    c21 = c2 == 1
    c20 = c2 == 0

    a = sum(c11 * c20)
    b = sum(c11 * c21)
    c = sum(c10 * c20)
    d = sum(c10 * c21)
    corr = np.cos(np.radians(180 / (1 + np.sqrt(b * c / a / d))))
    return corr


def correlations(data, continuous, binary):
    correlations = []

    for v1, v2 in itertools.combinations(binary, 2):
        try:
            correlations.append([v1, v2, tetrachoric(data[v1], data[v2]), None, 'tetrachoric'])
        except Exception as e:
            print(e)
            print(f'{v1}-{v2}: NaN')

    for b1, c2 in itertools.product(binary, continuous):
        try:
            correlations.append([b1, c2, pointbiserialr(data[b1], data[c2])[0],
                                 pointbiserialr(data[b1], data[c2])[1], 'pointbiserial'])
        except Exception as e:
            print(e)
            print(f'{b1}-{c2}: {e}')

    for c1, c2 in itertools.combinations(continuous, 2):
        try:
            correlations.append([c1, c2, pearsonr(data[c1], data[c2])[0],
                                 pearsonr(data[c1], data[c2])[1], 'pearsons'])
        except Exception as e:
            print(f'{c1}-{c2}: {e}')

    data_corr = pd.DataFrame.from_records(correlations, columns=['cov 1', 'cov 2', 'corr', 'p', 'type'])
    return data_corr
