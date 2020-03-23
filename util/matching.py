from scipy.spatial.distance import pdist, squareform
# import networkx as nx
# from networkx.algorithms.matching import max_weight_matching as mwm
import numpy as np
import pandas as pd
from typing import Union
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import statsmodels.formula.api as smf
import statsmodels.api as sm


class CEM:
    ''' Coarsened Exact Matching '''

    def __init__(self, data, treatment, outcome):
        self.data = data
        self.outcome = outcome
        self.treatment = treatment
        self.matched = None

    def match(self, bins):
        ''' Return matched data given a coursening '''
        # coarsen based on supplied bins
        coarsened = self.coarsen(self.data, bins)
        # filter groups with only control or treatment examples
        gb = list(self.data.drop([self.outcome, self.treatment], axis=1).columns.values)
        self.matched = coarsened.groupby(gb).filter(lambda x: len(x[self.treatment].unique()) > 1)
        # weight samples
        if len(self.matched):
            vc = self.matched[self.treatment].value_counts()
            mT, mC = vc[1], vc[0]
            weights = self.matched.groupby(gb).apply(lambda x: self.weight(x, mT, mC))['weight']
            self.matched = self.data.loc[self.matched.index, :]
            self.matched['weight'] = weights
        return self.matched.copy()

    def bins_gen(self, d):
        ''' Individual coarsening dict generator '''
        from itertools import product
        keys, values = d.keys(), d.values()
        combinations = product(*values)
        for c in combinations:
            yield dict(zip(keys, c))

    def relax(self, bins):
        ''' Match on several coarsenings and evaluate L1 metric '''
        from tqdm import tqdm
        rows = []
        length = np.prod([len(x) for x in bins.values()])
        for bins_i in tqdm(self.bins_gen(bins), total=length):
            matched = self.match(bins_i)
            if len(matched):
                L1 = self.score(matched.drop('weight', axis=1))
                vc = matched[self.treatment].value_counts()
                rows.append([L1, vc[0], vc[1], bins_i])
            else:
                rows.append([1, None, None, bins_i])
        results = pd.DataFrame.from_records(
            rows, columns=['statistic', 'controls', 'treatments', 'coarsening'])
        return results

    def regress(self, bins=None):
        if bins is None:
            assert self.matched is not None, 'Match data via coursening first.'
            return self._regress_matched()
        else:
            n_lists = sum(isinstance(x, (list, tuple)) for x in bins.values())
            if n_lists > 1:
                raise NotImplementedError('Cant handle depth>1 regression yet.')
            elif n_lists == 1:
                # Regress at different coarsenings
                raise NotImplementedError('TODO')
            else:
                self.match(bins)
                return self._regress_matched()

    def _regress_matched(self):
        right = ' + '.join(self.matched.drop([self.outcome, 'weight'], axis=1).columns.values)
        formula = f'{self.outcome} ~ {right}'
        glm = smf.glm(formula, data=self.matched, family=sm.families.Binomial(),
                      var_weights=self.matched['weight'])
        result = glm.fit(method='newton')
        return result.summary()

    def coarsen(self, data, bins):
        ''' Coarsen data based on schema '''
        df_coarse = data.apply(lambda x: pd.cut(x, bins=bins[x.name], labels=False), axis=0)
        return df_coarse

    def weight(self, group, mT, mC):
        ''' Calculate weights for regression '''
        vc = group[self.treatment].value_counts()
        mTs, mCs = vc[1], vc[0]
        group['weight'] = pd.Series([1 if x[self.treatment] else (mC / mT) * (mTs / mCs)
                                     for _, x in group.iterrows()], index=group.index)
        return group

    def score(self, data):
        ''' Evaluate histogram similarity '''
        treatments = data[self.treatment] == 1
        controls = data[self.treatment] == 0
        d = data.drop([self.outcome, self.treatment], axis=1)
        cont_bins = 4
        bins = [min(len(x.unique()), cont_bins) for _, x in d.items()]
        ranges = [(x.min(), x.max()) for _, x in d.items()]
        try:
            ht, _ = np.histogramdd(d.loc[treatments, :].to_numpy(), bins=bins, range=ranges, density=False)
            hc, _ = np.histogramdd(d.loc[controls, :].to_numpy(), bins=bins, range=ranges, density=False)
            L1 = np.sum(np.abs(ht / len(treatments) - hc / len(controls))) / 2
        except Exception as e:
            print(e)
            return 1
        return L1


def mahalanobis_frontier(data, on):
    print('Distances..', end=' ')
    distances = get_distances(data, 'mahalanobis', on)
    MMD_tc = distances.min(axis=0)  # gives closest control to each treatment
    MMD_ct = distances.min(axis=1)  # gives closest treatment to each control
    MMD = pd.concat([MMD_tc, MMD_ct])
    print('ready.')

    pruned_controls = []
    pruned_treatments = []
    AMD_ = []
    radii = []
    unique_distances = sorted(set(np.ravel(distances.round(1).to_numpy())), reverse=True)
    for radius in tqdm(unique_distances):
        AMD = MMD[MMD <= radius].mean()  # average min distance below threshold
        AMD_.append(AMD)
        treatments, controls = prune_by_distance(distances, radius)  # controls that werent pruned
        pruned_controls.append(len(distances) - len(controls))
        pruned_treatments.append(len(distances.columns) - len(treatments))
        radii.append(radius)

        # stopping criterion
        if pruned_controls[-1] == len(distances) or AMD_[-1] == 0:
            break

    return pd.DataFrame.from_dict({'pruned controls': pruned_controls, 'pruned treatments': pruned_treatments, 'AMD': AMD_, 'radius': radii}, orient='columns')


def prune_by_distance(distances, caliper: Union[int, float] = 5):
    matches = distances.le(caliper)
    # controls that have a treatment match
    control_idx = [idx for idx in matches.index if matches.loc[idx, :].sum()]
    # treatments that have a control match
    treatment_idx = [idx for idx in matches.columns if matches[idx].sum()]
    return treatment_idx, control_idx


def get_distances(data, distance, t, dtype=float):
    treatment, control = list(map(lambda x: x.index, split(data, t)))
    distances = pdist(data.drop(t, axis=1).astype(dtype), distance)
    distances = pd.DataFrame._from_arrays(squareform(distances), index=data.index, columns=data.index.values)
    distances = distances.loc[control, treatment.values]
    distances.index.rename('control', inplace=True)
    distances.columns.rename('treatment', inplace=True)
    return distances


def match_by_distance(data, on, dv, distance, caliper):
    if caliper == 'auto':
        df = mahalanobis_frontier(data.drop(dv, axis=1), on)
        df['AMD'] = (df['AMD'] - df['AMD'].min()) / (df['AMD'].max() - df['AMD'].min())
        df['pruned controls'] = (df['pruned controls'] - df['pruned controls'].min()) / \
            (df['pruned controls'].max() - df['pruned controls'].min())
        df['dto'] = np.sqrt(df['AMD']**2 + df['pruned controls']**2)
        caliper = df.loc[df['dto'] == df['dto'].min(), 'radius']

    distances = get_distances(data.drop(dv, axis=1), distance, on)
    treatments, controls = prune_by_distance(distances, caliper)
    df_matched = data.loc[controls + treatments, :]
    return df_matched


def split(data, on):
    return data.loc[data[on], :], data.loc[~data[on], :],


def covariate_dists(data, on, kde=True, hist=True, n_bins=10):
    vals = data[on].unique()
    flatui = ["#2ecc71", "#9b59b6", "#3498db", "#e74c3c", "#34495e"]

    for col in data.drop(on, axis=1).columns:
        bins = np.linspace(data[col].min(), data[col].max(), n_bins) if hist else None
        try:
            for i, val in enumerate(vals):
                sns.distplot(data[data[on] == val][col],
                             bins=bins, label=f'{on}={val}', kde=kde, norm_hist=hist, hist=hist, color=flatui[i])
                plt.axvline(data[data[on] == val][col].mean(), color=flatui[i])
        except:
            for i, val in enumerate(vals):
                sns.distplot(data[data[on] == val][col],
                             bins=bins, label=f'{on}={val}', kde=False, norm_hist=True, hist=True, color=flatui[i])
                plt.axvline(data[data[on] == val][col].mean(), color=flatui[i])

        plt.title(f'{col} distributions')
        plt.legend()
        plt.show()


def dist_test(data, on, func, **kwargs):
    v = data[on].unique()
    if len(v) != 2:
        print('Dichotamize variable of interest and try again.')
        return None

    res = []
    for col in data.drop(on, axis=1).columns:
        if len(data[col].unique()) <= 2:
            res.append([np.nan])
        else:
            res.append(func(data.loc[:, [on, col]], on))

    return pd.DataFrame.from_records(res, index=data.drop(on, axis=1).columns, columns=['statistic'])


def binom_z_tests(data, on):
    from scipy.stats import norm

    v = data[on].unique()
    if len(v) != 2:
        print(f'Dichotamize variable of interest and try again. {len(v)}')
        return None

    res = []
    for col in data.drop(on, axis=1).columns:
        if len(data[col].unique()) != 2:
            res.append([np.nan])
        else:
            p1 = data[data[on] == v[0]][col].mean()
            p2 = data[data[on] == v[1]][col].mean()
            n1 = len(data[data[on] == v[0]][col])
            n2 = len(data[data[on] == v[1]][col])
            p = (n1 * p1 + n2 * p2) / (n1 + n2)
            z = (p1 - p2) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
            res.append([z, norm.sf(abs(z)) * 2])

    return pd.DataFrame.from_records(res, index=data.drop(on, axis=1).columns, columns=['statistic'])
