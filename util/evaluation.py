import itertools
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from scipy.stats import norm


def to_frame(summary: 'statsmodels.Summary') -> pd.DataFrame:
    '''Convert Summary object to DataFrame of covariate regression statistics'''
    pd_summary = pd.read_html(summary.tables[1].as_html())[0]
    pd_summary.iloc[0, 0] = 'covariate'
    pd_summary.columns = pd_summary.iloc[0]
    pd_summary = pd_summary[1:]
    pd_summary.set_index('covariate', inplace=True)
    return pd_summary.astype(float)


def residuals(group, sd=2):
    avg_pred = group['pred'].mean()
    avg_resid = group['residual'].mean()
    bound = sd * np.sqrt(avg_pred * (1 - avg_pred) / len(group))
    df = pd.DataFrame.from_dict({'pred': [avg_pred], 'residual': [avg_resid], 'bound': [bound], 'count': [
                                len(group)], 'inside': (avg_resid <= bound) and (avg_resid >= -bound)}, orient='columns')
    for c in group.drop(['pred', 'residual'], axis=1).columns:
        try:
            df[c] = group[c].mean()
        except Exception as e:
            df[c] = np.nan
    return df


def binned_residuals(estimator, X, y, bins=None, on='pred'):
    y_pred = estimator.predict(X)
    y_ = pd.DataFrame.from_dict({'pred': y_pred, 'true': y, 'residual': y - y_pred}, orient='columns')
    for c in X.columns:
        y_[c] = X[c]
    if bins is None:
        bins = int(np.sqrt(len(y_)))
    print(f'Using {bins} bins.')
    y_['bin'] = pd.qcut(y_[on], q=bins, duplicates='drop')
    y_.reset_index(drop=True, inplace=True)
    y_resid = y_.groupby('bin').apply(residuals)
    y_resid.index = y_resid.index.droplevel(-1)
    return y_resid


def plot_binned_residuals(summary, lims=None, on='pred'):
    summary['lower'] = -summary['bound']
    summary['upper'] = summary['bound']
    cmap = {True: 'g', False: 'r'}
    ax = sns.scatterplot(x=on, y='residual', data=summary, hue='inside',
                         style='inside', palette=cmap, legend=False)
    ax = sns.lineplot(x=on, y='lower', data=summary, ax=ax, c='b')
    ax = sns.lineplot(x=on, y='upper', data=summary, ax=ax, c='b')

    if lims is not None:
        plt.xlim(lims[0], lims[1])
    plt.ylabel('Average residual')
    plt.xlabel(f'Average {on}')
    plt.title(f'Binned residuals for {on}')
    return ax


def odds_ratios(g, treatment, hue='distance', hue_values=35, p=0.1, plot=True):
    rows = []
    if isinstance(hue_values, int):
        vals = [hue_values]
    else:
        vals = hue_values
    for v in vals:
        base = odds(g, p, **{treatment: False, hue: v})
        odds_Xi = odds(g, p, **{treatment: True, hue: v})
        OR = odds_Xi / base
        row = {'OR': OR, f'{hue}': v}
        rows.append(row)
    res = pd.DataFrame.from_records(rows)
    if plot:
        ax = sns.lineplot(x=hue, y='OR', data=res)
        ax.set_xlabel(hue)
        ax.set_ylabel('Odds Ratio')
        ax.set_title(f'Odds ratios for {treatment} at varying {hue}')
        return res, ax
    return res


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
