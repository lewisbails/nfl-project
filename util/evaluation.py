import itertools
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def odds(coefficients: pd.Series, **kwargs):
    ''' Added odds thanks to these coefficients and theyre values '''

    # becomes odds ratio if the alternate would result in np.exp(..)=1
    # print(kwargs, end='')
    sum_ = 0
    # main
    for arg, val in kwargs.items():
        try:
            sum_ += coefficients[arg] * val
        except:
            pass
    # interactions
    for i, j in itertools.product(kwargs.keys(), kwargs.keys()):
        if i > j:
            try:
                sum_ += coefficients[f'{i}:{j}'] * kwargs[i] * kwargs[j]
            except:
                try:
                    sum_ += coefficients[f'{j}:{i}'] * kwargs[i] * kwargs[j]
                except:
                    pass

    # print(sum_)

    return np.exp(sum_)


def confusion(covs: list, coefficients: pd.Series):
    ''' Confusion matrix of odd ratios. Base odds NOT chose by user. '''

    assert len(covs) == 2, 'Can only plot 2D confusion matrix.'
    cols = {}
    ranges = []
    bases = []
    for i, cov in enumerate(covs):
        if covs[i] == 'dist':
            ranges.append(range(20, 65, 5))
            bases.append(20)
        elif covs[i] == 'seasons':
            ranges.append(range(1, 26, 3))
            bases.append(1)
        else:
            ranges.append(range(2))
            bases.append(0)

    base_odds = odds(coefficients, **{covs[0]: bases[0], covs[1]: bases[1]})
    for i in ranges[0]:  # build cols
        rows = []
        for j in ranges[1]:
            rows.append(odds(coefficients, **{covs[0]: i, covs[1]: j}) / base_odds)
        cols[covs[0] + str(i)] = pd.Series(rows, index=[covs[1] + str(k) for k in ranges[1]])
    df = pd.DataFrame.from_dict(cols, orient='columns')
    ax = sns.heatmap(df, annot=True)
    ax.set_title(f' Odds Ratios: {covs[0]} vs {covs[1]}')
    plt.show()
    return ax
