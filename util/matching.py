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
from copy import deepcopy


class CEM:
    ''' Coarsened Exact Matching '''

    @staticmethod
    def match(data, treatment, bins, one_to_many=True):
        ''' Return matched data given a coursening '''
        # coarsen based on supplied bins
        data_ = data.copy()
        data_ = CEM.coarsen(data_, bins)

        # only keep stata with examples from each treatment level
        gb = list(data_.drop(treatment, axis=1).columns.values)
        matched = data_.groupby(gb).filter(lambda x: len(
            x[treatment].unique()) == len(data_[treatment].unique()))

        weights = pd.Series([0] * len(data_), index=data_.index)
        if len(matched) and one_to_many:
            weights = CEM.weight(matched, treatment, weights)
        else:
            pass
            # TODO: 1:1 matching using bhattacharya for each stratum, weight is 1 for the control and its treatment pair
        return weights

    @staticmethod
    def weight(data, treatment, initial_weights=None):
        if initial_weights is None:
            initial_weights = pd.Series([0] * len(data), index=data.index)
        counts = data[treatment].value_counts()
        gb = list(data.drop(treatment, axis=1).columns.values)
        weights = data.groupby(gb)[treatment].transform(lambda x: CEM._weight_stratum(x, counts))
        return weights.add(initial_weights, fill_value=0)

    @staticmethod
    def _weight_stratum(stratum, M):
        ''' Calculate weights for regression '''
        ms = stratum.value_counts()
        T = stratum.max()  # use as "treatment"
        return pd.Series([1 if c == T else (M[c] / M[T]) * (ms[T] / ms[c]) for _, c in stratum.iteritems()])

    @staticmethod
    def _bins_gen(d):
        ''' Individual coarsening dict generator '''
        from itertools import product
        from collections import OrderedDict
        od = OrderedDict(d)
        covariate, values = od.keys(), od.values()
        cut_types = [v['cut'] for v in values]
        bins = [v['bins'] for v in values]
        for bin_ in bins:
            if not isinstance(bin_, (range)):
                raise TypeError('Ambiguous relax process. Please use ranges.')
        combinations = product(*bins)
        for c in combinations:
            dd = [(i, {'bins': j, 'cut': k}) for i, j, k in zip(covariate, c, cut_types)]
            yield dict(dd)

    @staticmethod
    def relax(data, treatment, bins):
        ''' Match on several coarsenings and evaluate L1 metric '''
        from tqdm import tqdm
        data_ = data.copy()
        rows = []
        length = np.prod([len(x['bins']) for x in bins.values()])
        for bins_i in tqdm(CEM._bins_gen(bins), total=length):
            weights = CEM.match(data_, treatment, bins_i)
            if (weights > 0).sum():
                d = data_.loc[weights > 0, :]
                if treatment in bins_i:
                    d[treatment] = CEM._cut(d[treatment], **bins_i[treatment])
                L1 = CEM.score(d, treatment)
                vc = d[treatment].value_counts()
                rows.append([L1, vc, bins_i])
            else:
                rows.append([1, None, bins_i])
        results = pd.DataFrame.from_records(
            rows, columns=['statistic', 'counts', 'coarsening'])
        return results

    @staticmethod
    def regress(data, treatment, outcome, bins, drop=[]):
        data_ = data.copy()
        bins_ = deepcopy(bins)
        n_lists = sum(isinstance(x['bins'], range) for x in bins_.values())
        if n_lists > 1:
            raise NotImplementedError('Cant handle depth>1 regression yet.')
        elif n_lists == 1:
            # Regress at different coarsenings
            results = {}
            k = list(filter(lambda k: isinstance(bins_[k]['bins'], range), bins_))[0]
            v = bins_[k]['bins']
            method = bins_[k]['cut']
            print(f'Regressing with {len(v)} different pd.{method} binnings on "{k}"\n')
            for i in tqdm(v):
                bins_[k].update({'bins': i})
                weights_ = CEM.match(data_.drop(outcome, axis=1), treatment, bins_)

                # scoring
                d_score = data_.drop(outcome, axis=1).loc[weights_ > 0, :]
                if treatment in bins_:
                    d_score[treatment] = CEM._cut(d_score[treatment], **bins_i[treatment])
                L1 = CEM.score(d_score, treatment)

                # Weighted regression
                summary = CEM._regress_matched(data_, outcome, weights_, drop)

                results[i] = {'summary': summary,
                              'L1': L1,
                              'n_bins': i,
                              'var': k,
                              'vc': data_.loc[weights_ > 0, treatment].value_counts()}
            return results
        else:
            weights_ = CEM.match(data_.drop(outcome, axis=1), treatment, bins)
            return CEM._regress_matched(data_, outcome, weights_, drop)

    @staticmethod
    def _regress_matched(data, outcome, weights, drop):
        weights_ = weights[weights > 0]
        data_ = data.loc[weights_.index, :].copy()

        right = ' + '.join(data_.drop([outcome] + drop, axis=1).columns.values)
        formula = f'{outcome} ~ {right}'
        glm = smf.glm(formula, data=data_, family=sm.families.Binomial(),
                      var_weights=weights_)
        result = glm.fit(method='bfgs')
        return result.summary()

    @staticmethod
    def _cut(col, method, bins):
        if method == 'qcut':
            return pd.qcut(col, q=bins, labels=False)
        elif method == 'cut':
            return pd.cut(col, bins=bins, labels=False)
        else:
            raise Exception(f'Pandas cut method "{method}" not supported.')

    @staticmethod
    def coarsen(data, bins):
        ''' Coarsen data based on schema '''
        df_coarse = data.apply(lambda x: CEM._cut(
            x, bins[x.name]['cut'], bins[x.name]['bins']) if x.name in bins else x, axis=0)
        return df_coarse

    @staticmethod
    def score(data, treatment):
        ''' Evaluate histogram similarity '''
        from itertools import combinations
        from collections import OrderedDict
        groups = data.groupby(treatment).groups
        data_ = data.drop(treatment, axis=1).copy()
        cont_bins = 4
        bins = [min(len(x.unique()), cont_bins) for _, x in data_.items()]
        ranges = [(x.min(), x.max()) for _, x in data_.items()]
        try:
            h = {}
            for k, i in groups.items():
                h[k] = np.histogramdd(data_.loc[i, :].to_numpy(), bins=bins, range=ranges, density=False)[0]
            L1 = {}
            for pair in map(dict, combinations(h.items(), 2)):
                pair = OrderedDict(pair)
                (k_left, k_right), (h_left, h_right) = pair.keys(), pair.values()  # 2 keys 2 histograms
                L1[tuple([k_left, k_right])] = np.sum(
                    np.abs(h_left / len(groups[k_left]) - h_right / len(groups[k_right]))) / 2
        except Exception as e:
            print(e)
            return 1
        if len(L1) == 1:
            return list(L1.values())[0]
        return L1

    def LSATT(self):
        # only currently valid for dichotamous treatments
        from scipy.stats import ttest_ind_from_stats
        from functools import reduce
        from collections import OrderedDict
        df2 = pd.concat((self.data, self.weights.rename('weights')), axis=1)
        df2 = df2.loc[df2['weights'] > 0, :]
        res = OrderedDict()
        for i, g in df2.groupby(self.treatment):
            weight = g['weights'].sum()

            WSOUT = (g[self.outcome] * g['weights']).sum()
            wave = WSOUT / weight

            ave = g[self.outcome].mean()
            WSSR = (g['weights'] * (g[self.outcome] - ave)**2).sum()
            wstd = np.sqrt(WSSR / weight)

            res[i] = [wave, wstd, len(g)]

        return res, ttest_ind_from_stats(*list(reduce(lambda x, y: x + y, res.values())), equal_var=False)


def summary_to_frame(summary, n_bins, vc, statistic, dtype=None):
    observations = pd.read_html(summary.tables[0].as_html())[0].iloc[0, 3]
    results = pd.read_html(summary.tables[1].as_html())[0]
    results.iloc[0, 0] = 'covariate'
    results.columns = results.iloc[0]
    results = results[1:]
    results['observations'] = observations
    results['n_bins'] = n_bins
    results['controls'] = vc[False]
    results['treatments'] = vc[True]
    results['coef'] = results['coef'].astype(float)
    results['P>|z|'] = results['P>|z|'].astype(float)
    results['statistic'] = statistic
    if dtype:
        for c in results.columns:
            if results[c].dtype not in (object, bool):
                results[c] = results[c].astype(dtype)
    return results


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
