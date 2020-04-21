import itertools
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from scipy.stats import norm


def odds(data, p=1, **kwargs):
    ''' Added odds thanks to these coefficients and theyre values '''
    kwargs.update({'np': np})
    total = 0
    for cov, row in data.iterrows():
        cov_ = cov.replace(':', '*')
        try:
            val = eval(cov_, kwargs)
            sig = row['P>|z|'] <= p
            coef = row['coef']
            total += val * sig * coef
        except:
            continue
    return np.exp(total)


def confusion(covs: list, coefficients: pd.Series):
    ''' Confusion matrix of odd ratios. Base odds NOT chose by user. '''

    assert len(covs) == 2, 'Can only plot 2D confusion matrix.'
    covs_ = {}
    for i, cov in enumerate(covs):
        if covs[i] == 'dist':
            covs_[i] = {'key': cov, 'range': range(20, 65, 5), 'base': 20}
        elif covs[i] == 'seasons':
            covs_[i] = {'key': cov, 'range': range(1, 26, 3), 'base': 1}
        elif covs[i] == 'pressure':
            covs_[i] = {'key': cov, 'range': range(0, 6, 1), 'base': 0}
        elif covs[i] == 'form':
            covs_[i] = {'key': cov, 'range': range(0, 1, 0.1), 'base': 0}
        elif covs[i] == 'temperature':
            covs_[i] = {'key': cov, 'range': range(30, 80, 5), 'base': 30}
        elif covs[i] == 'wind':
            covs_[i] = {'key': cov, 'range': range(0, 30, 5), 'base': 0}
        else:
            covs_[i] = {'key': cov, 'range': range(2), 'base': 0}

    l = covs_[0]['key']
    r = covs_[1]['key']
    cols = {}
    base_odds = odds(coefficients, **{l: covs_[0]['base'], r: covs_[1]['base']})
    for i in covs_[0]['range']:  # build cols
        rows = []
        for j in covs_[1]['range']:
            rows.append(odds(coefficients, **{l: i, r: j}) / base_odds)
        cols[f'{l}_{i}'] = pd.Series(rows, index=[f'{r}_{k}' for k in covs_[1]['range']])
    df = pd.DataFrame.from_dict(cols, orient='columns')
    ax = sns.heatmap(df, annot=True)
    ax.set_title(f'Odds Ratios: {l} vs {r}')
    plt.show()
    return ax
