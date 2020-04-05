import numpy as np
import pandas as pd
from pandas import Series
from pandas.core.frame import _from_nested_dict
from typing import Union
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import statsmodels.formula.api as smf
import statsmodels.api as sm
from copy import deepcopy
from itertools import combinations, product
from collections import OrderedDict
from scipy.stats import ttest_ind_from_stats
from scipy.stats import ttest_ind
from functools import reduce
from scipy.stats import chisquare
import copy


class CEM:

    def __init__(self, data, treatment, outcome, continuous=[], H=None):
        self.data = data.copy()
        self.treatment = treatment
        self.outcome = outcome
        self.continuous = continuous
        self.H = H
        self.bins = None
        self.ranges = None
        self.preimbalance = None

        #  find H, get bins and ranges
        df = self.data.drop(outcome, axis=1)
        if self.H is None:
            print('Calculating H, this may take a few minutes.')
            rows = []
            cont_bins = range(1, 10)
            imb = []
            for h in cont_bins:
                # we use H bins for continuous variables when evaluating imbalance
                l1 = imbalance(df, self.treatment, continuous=self.continuous, H=h)
                imb.append(l1)
            imb = pd.Series(imb, index=cont_bins)
            self.H = (imb.sort_values(ascending=False) <= imb.quantile(.5)).idxmax()

        self.preimbalance, (self.bins, self.ranges) = imbalance(
            df, self.treatment, continuous=self.continuous, H=self.H, retargs=True)

    def imbalance(self, coarsening, measure='l1', one_to_many=True):
        weights = self.match(coarsening, one_to_many)
        df = self.data.drop(self.outcome, axis=1).loc[weights > 0, :]
        return imbalance(df, self.treatment, measure, bins=self.bins, ranges=self.ranges)

    def univariate_imbalance(self, coarsening, measure='l1', one_to_many=True):
        weights = self.match(coarsening, one_to_many)
        df = self.data.drop(self.outcome, axis=1).loc[weights > 0, :]
        return UnivariateBalance(df, self.treatment, measure, bins=self.bins, ranges=self.ranges)

    def match(self, coarsening, one_to_many=True):
        return match(self.data.drop(self.outcome, axis=1), self.treatment, coarsening, one_to_many)

    def relax(self, coarsening, relax_vars, measure='l1'):
        return Relax(self.data.drop(self.outcome, axis=1), self.treatment, coarsening, relax_vars, measure, self.continuous,
                     bins=self.bins, ranges=self.ranges)

    def regress(self, coarsening, relax_vars=[], measure='l1', formula=None, drop=[]):
        return Regress(self.data, self.treatment, self.outcome, coarsening,
                       relax_vars=relax_vars, measure=measure, formula=formula,
                       drop=drop, continuous=self.continuous,
                       bins=self.bins, ranges=self.ranges)

    def LSATT(self, coarsening, one_to_many=True):
        weights = self.match(coarsening, one_to_many)
        return LSATT(self.data, self.treatment, self.outcome, weights)


class UnivariateBalance:

    def __init__(self, data, treatment, measure='l1', bins=None, ranges=None):
        assert len(data.drop(treatment, axis=1).columns) == len(
            bins) == len(ranges), 'Lengths not equal.'
        if measure != 'l1':
            raise NotImplementedError('Only L1 possible at the moment.')
        self.data = data
        self.treatment = treatment
        self.bins = bins
        self.ranges = ranges
        self.measure = measure
        self.summary = self._summarise()

    def _summarise(self):
        marginal = {}
        for col, bin_, range_ in zip(self.data.drop(self.treatment, axis=1).columns, self.bins, self.ranges):
            cem_imbalance = imbalance(self.data.loc[:, [col, self.treatment]],
                                      self.treatment, self.measure, bins=[bin_], ranges=[range_])
            d_treatment = self.data.loc[self.data[self.treatment] > 0, col]
            d_control = self.data.loc[self.data[self.treatment] == 0, col]
            if self.data[col].nunique() > 2:
                stat = d_treatment.mean() - d_control.mean()
                _, p = ttest_ind(d_treatment, d_control, equal_var=False)
                type_ = 'diff'
            else:
                f_obs = d_treatment.value_counts(normalize=True)
                f_exp = d_control.value_counts(normalize=True)
                stat, p = chisquare(f_obs, f_exp)
                type_ = 'Chi2'

            q = [0, 0.25, 0.5, 0.75, 1]
            diffs = d_treatment.quantile(
                q) - d_control.quantile(q) if type_ == 'diff' else pd.Series([None] * 5, index=q)
            row = {'imbalance': cem_imbalance, 'measure': self.measure,
                   'statistic': stat, 'type': type_, 'P>|z|': p}
            row.update({f'{int(i*100)}%': diffs[i] for i in q})
            marginal[col] = pd.Series(row)
        return pd.DataFrame.from_dict(marginal, orient='index')

    def plot(self, kde=False, hist=True):
        vals = self.data[self.treatment].unique()
        flatui = ["#2ecc71", "#9b59b6", "#3498db", "#e74c3c", "#34495e"]

        for col, bin_, range_ in zip(self.data.drop(self.treatment, axis=1).columns, self.bins, self.ranges):
            try:
                for i, val in enumerate(vals):
                    sns.distplot(self.data[self.data[self.treatment] == val][col],
                                 bins=bin_, hist_kws={'range': range_}, label=f'{self.treatment}={val}', kde=kde, norm_hist=hist, hist=hist, color=flatui[i])
                    plt.axvline(self.data[self.data[self.treatment] == val][col].mean(), color=flatui[i])
            except:
                for i, val in enumerate(vals):
                    sns.distplot(self.data[self.data[self.treatment] == val][col],
                                 bins=bin_, hist_kws={'range': range_}, label=f'{self.treatment}={val}', kde=False, norm_hist=True, hist=True, color=flatui[i])
                    plt.axvline(self.data[self.data[self.treatment] == val][col].mean(), color=flatui[i])

            plt.title(f'{col} distributions')
            plt.legend()
            plt.show()


class Regress:
    def __init__(self, data, treatment, outcome, coarsening, **kwargs):
        self.data = data
        self.treatment = treatment
        self.outcome = outcome
        self.coarsening = coarsening
        self.__dict__.update(**kwargs)
        self.summary = regress(data, treatment, outcome, coarsening, **kwargs)
        if isinstance(self.summary, pd.Series):
            self.summary = pd.DataFrame([self.summary])

    def _sm_summary_to_frame(self, summary):
        pd_summary = pd.read_html(summary.tables[1].as_html())[0]
        pd_summary.iloc[0, 0] = 'covariate'
        pd_summary.columns = pd_summary.iloc[0]
        pd_summary = pd_summary[1:]
        pd_summary.set_index('covariate', inplace=True)
        return pd_summary.astype(float)

    def _get_sample_info(self, row):
        sample_info = {}
        sample_info['imbalance'] = row['imbalance']
        sample_info['observations'] = pd.read_html(row['result'].summary().tables[0].as_html())[0].iloc[0, 3]
        for i, v in row['vc'].items():
            sample_info[f'treatment_{i}'] = v
        return pd.Series(sample_info)

    def _row_to_long(self, row):
        summary = self._sm_summary_to_frame(row['result'].summary())
        summary['var'] = row['var']
        summary['n_bins'] = row.name if 'n_bins' not in row.index else row['n_bins']
        summary.set_index(['n_bins', 'var'], append=True, inplace=True)
        return summary

    def covariates(self):
        return pd.concat([self._row_to_long(row) for _, row in self.summary.iterrows()])

    def coarsenings(self):
        return self.summary.apply(self._get_sample_info, axis=1)

    def _lineplot(self, data, ax):
        colours = {}
        for i, g in data.groupby('covariate'):
            g.plot.line(x='n_bins', y='coef', ax=ax, label=i)
            c = plt.gca().lines[-1].get_color()
            colours[i] = c
        return ax, colours

    def _scatterplot(self, data, ax, colours=None, stars=True):
        for i, g in data.groupby('covariate'):
            g.plot.scatter(x='n_bins', y='coef', s=g['P>|z|'] * 100,
                           c=[colours[i]] if colours else None, ax=ax, label=i)
            if stars:
                for j, row in g.iterrows():
                    if row['P>|z|'] <= 0.01:
                        txt = '***'
                    elif row['P>|z|'] <= 0.05:
                        txt = '**'
                    elif row['P>|z|'] <= 0.1:
                        txt = '*'
                    else:
                        txt = ''
                    ax.text(row['n_bins'], row['coef'], txt, fontsize=16)
        return ax

    def plot(self, stars=True):
        lf = self.covariates().reset_index()
        if lf['var'].nunique() > 1:
            raise Exception('Progressive coarsening plot only available for single variable.')
        else:
            var = lf['var'].iloc[0]

        fig, ax = plt.subplots()
        r = lf.reset_index()
        r = r.loc[r['covariate'] != 'Intercept', :]

        ax, colours = self._lineplot(r, ax)
        line_leg = ax.legend(loc='upper left', title='Covariates', bbox_to_anchor=(1.05, 1))
        for line in line_leg.get_lines():
            line.set_linewidth(4.0)

        ax = self._scatterplot(r, ax, colours, stars)

        from matplotlib.lines import Line2D
        sizes = np.array([1, 5, 10, 20, 50])
        circles = [Line2D([0], [0], linewidth=0.01, marker='o', color='w', markeredgecolor='g',
                          markerfacecolor='g', markersize=np.sqrt(size)) for size in sizes]
        scatter_leg = ax.legend(circles, sizes / 100, loc='lower left',
                                title='P-values', bbox_to_anchor=(1.05, 0))
        ax.add_artist(line_leg)

        fig.set_size_inches(12, 8)
        ax.set_title('Regression coefficients for progressive coarsening.')
        ax.set_ylabel('Coefficient')
        ax.set_xlabel(f'# bins for {var} coarsening')
        return ax


class Relax:
    '''Returned from calling relax on the CEM instance'''

    def __init__(self, data, treatment, coarsening, relax_vars, measure='l1', continuous=[], **kwargs):
        self.total = len(data)
        self.relax_vars = relax_vars
        self.coarsening = coarsening
        self.measure = measure
        self.continuous = continuous
        self.__dict__.update(**kwargs)
        self.summary = relax(data, treatment, coarsening, relax_vars, measure, continuous, **kwargs)

    def plot(self, **kwargs):
        s = self.summary.copy()
        t_cols = [col for col in self.summary.columns if 'treatment' in col]
        s['# matched'] = s.loc[:, t_cols].sum(axis=1)
        s['% matched'] = (s['# matched'] / self.total * 100).round(1)
        if len(self.relax_vars) == 1:
            var = self.relax_vars[0][0]
            x = f'{var} bins'
            s[x] = [i[var]['bins'] for i in s['coarsening']]
        else:
            x = 'coarsening'
            s[x] = range(len(s))
            s = s.sort_values('% matched')
        ax = sns.lineplot(x=x, y='% matched', data=s, style='measure', markers=True, **kwargs)
        ax.set_title('Multivariate L1 for progressive coarsening')
        for _, row in s.iterrows():
            ax.text(row[x] + 0.1, row['% matched'] + 0.1, round(row['imbalance'], 2), fontsize=10)
        return ax, s


# The user can use all the functions outside of the classes if they choose to

def match(data, treatment, bins, one_to_many=True):
    ''' Return weights for data given a coursening '''
    # coarsen based on supplied bins
    data_ = coarsen(data.copy(), bins)

    # only keep stata with examples from each treatment level
    gb = list(data_.drop(treatment, axis=1).columns.values)
    matched = data_.groupby(gb).filter(lambda x: len(
        x[treatment].unique()) == len(data_[treatment].unique()))

    # weight data in surviving strata
    weights = pd.Series([0] * len(data_), index=data_.index)
    if len(matched) and one_to_many:
        weights = weight(matched, treatment, weights)
    else:
        raise NotImplementedError
        # TODO: k:k matching using bhattacharya for each stratum, weight is 1 for the control and its treatment pair
    return weights


def weight(data, treatment, initial_weights=None):
    if initial_weights is None:
        initial_weights = pd.Series([0] * len(data), index=data.index)
    counts = data[treatment].value_counts()
    gb = list(data.drop(treatment, axis=1).columns.values)
    weights = data.groupby(gb)[treatment].transform(lambda x: _weight_stratum(x, counts))
    return weights.add(initial_weights, fill_value=0)


def _weight_stratum(stratum, M):
    ''' Calculate weights for regression '''
    ms = stratum.value_counts()
    T = stratum.max()  # use as "treatment"
    return pd.Series([1 if c == T else (M[c] / M[T]) * (ms[T] / ms[c]) for _, c in stratum.iteritems()])


def _bins_gen(d, relax_on):
    ''' Individual coarsening dict generator '''
    name = [v[0] for v in relax_on]
    cut_types = [v[-1] for v in relax_on]
    bins = [v[1] for v in relax_on]
    combinations = product(*bins)
    for c in combinations:
        dd = copy.deepcopy(d)
        new = {i: {'bins': j, 'cut': k} for i, j, k in zip(name, c, cut_types)}
        dd.update(new)
        yield dd


def relax(data, treatment, coarsening, relax_vars, measure='l1', continuous=[], **kwargs):
    ''' Match on several coarsenings and evaluate some imbalance measure '''
    assert all([len(x) == 3 for x in relax_vars]
               ), 'Expected variables to relax on as tuple triples (name, iterable, cut method)'
    data_ = data.copy()
    length = np.prod([len(x[1]) for x in relax_vars])

    if 'bins' not in kwargs or 'ranges' not in kwargs:
        kwargs = get_imbalance_params(data_.drop(
            treatment, axis=1), measure, continuous=continuous)  # indep. of any coarsening

    rows = []
    for coarsening_i in tqdm(_bins_gen(coarsening, relax_vars), total=length):
        weights = match(data_, treatment, coarsening_i)
        nbins = np.prod([x['bins'] if isinstance(x['bins'], int) else len(x['bins']) - 1
                         for x in coarsening_i.values()])
        if (weights > 0).sum():
            d = data_.loc[weights > 0, :]
            if treatment in coarsening_i:
                # continuous treatment binning
                d[treatment] = _cut(d[treatment], coarsening_i[treatment]
                                    ['cut'], coarsening_i[treatment]['bins'])
            score = imbalance(d, treatment, measure, **kwargs)
            vc = d[treatment].value_counts()
            row = {'imbalance': score, 'measure': measure, 'coarsening': coarsening_i, 'bins': nbins}
            row.update({f'treatment_{t}': c for t, c in vc.items()})
            rows.append(pd.Series(row))
        else:
            rows.append(pd.Series({'imbalance': 1, 'measure': measure,
                                   'coarsening': coarsening_i, 'bins': nbins}))
    return pd.DataFrame.from_records(rows)


def regress(data, treatment, outcome, coarsening, relax_vars=[], measure='l1', formula=None, drop=[], continuous=[], **kwargs):
    '''Regress on 1 or more coarsenings and return a summary and imbalance measure'''
    data_ = data.copy()
    coarsening_ = deepcopy(coarsening)

    if not formula:
        formula = _infer_formula(data_, outcome, drop)

    n_relax = len(relax_vars)
    if n_relax > 1:
        raise NotImplementedError('Cant handle depth>1 regression yet.')
    elif n_relax == 1:
        # Regress at different coarsenings
        k = relax_vars[0][0]
        v = relax_vars[0][1]
        method = relax_vars[0][2]
        rows = []
        print(f'Regressing with {len(v)} different pd.{method} binnings on "{k}"\n')
        for i in tqdm(v):
            coarsening_[k].update({'bins': i, 'method': method})
            row = regress(data_, treatment, outcome, coarsening_,
                          formula=formula, continuous=continuous, **kwargs)
            row['n_bins'] = i
            row['var'] = k
            rows.append(row)
        frame = pd.DataFrame.from_records(rows)
        frame.set_index('n_bins', inplace=True)
        return frame
    else:
        # weights
        weights_ = match(data_.drop(outcome, axis=1), treatment, coarsening_)
        # imbalance
        if 'bins' not in kwargs or 'ranges' not in kwargs:
            imb_params = get_imbalance_params(data_.drop(
                [treatment, outcome], axis=1), measure, continuous=continuous)  # indep. of any coarsening
        else:
            imb_params = kwargs.copy()
        d = data_.drop(outcome, axis=1).loc[weights_ > 0, :]
        if treatment in coarsening_:
            d[treatment] = _cut(d[treatment], coarsening_[treatment]['cut'], coarsening_[
                treatment]['bins'])  # labels will be ints
        score = imbalance(d, treatment, measure, **imb_params)
        # regression
        res = _regress_matched(data_, formula, weights_)
        # counts
        vc = d[treatment].value_counts()  # ints if cut_ else original values
        return pd.Series({'result': res, 'imbalance': score, 'vc': vc, 'coarsening': coarsening_})


def _regress_matched(data, formula, weights):
    glm = smf.glm(formula,
                  data=data.loc[weights > 0, :],
                  family=sm.families.Binomial(),
                  var_weights=weights[weights > 0])
    result = glm.fit(method='bfgs')
    return result


def _infer_formula(data, dv, drop):
    iv = ' + '.join(data.drop([dv] + drop, axis=1).columns.values)
    return f'{dv} ~ {iv}'


def _cut(col, method, bins):
    if method == 'qcut':
        return pd.qcut(col, q=bins, labels=False)
    elif method == 'cut':
        return pd.cut(col, bins=bins, labels=False)
    else:
        raise Exception(
            f'"{method}" not supported. Coarsening only possible with "cut" and "qcut".')


def coarsen(data, coarsening):
    ''' Coarsen data based on schema '''
    df_coarse = data.apply(lambda x: _cut(
        x, coarsening[x.name]['cut'], coarsening[x.name]['bins']) if x.name in coarsening else x, axis=0)
    return df_coarse


def imbalance(data, treatment, measure='l1', **kwargs):
    ''' Evaluate histogram similarity '''
    if measure in MEASURES:
        return MEASURES[measure](data, treatment, **kwargs)
    else:
        raise NotImplementedError(f'"{measure}" not a valid measure.')


def _L1(data, treatment, bins=None, ranges=None, retargs=False, continuous=[], H=5):

    groups = data.groupby(treatment).groups
    data_ = data.drop(treatment, axis=1).copy()

    if len(continuous):
        params = get_imbalance_params(data_, 'l1', continuous=continuous, H=H)
        bins, ranges = params['bins'], params['ranges']
    else:
        if bins is None or ranges is None:
            raise Exception('continuous parameter not supplied but neither are the bins and ranges.')

    try:
        h = {}
        for k, i in groups.items():
            h[k] = np.histogramdd(data_.loc[i, :].to_numpy(), density=False, bins=bins, range=ranges)[0]
        L1 = {}
        for pair in map(dict, combinations(h.items(), 2)):
            pair = OrderedDict(pair)
            (k_left, k_right), (h_left, h_right) = pair.keys(), pair.values()  # 2 keys 2 histograms
            L1[tuple([k_left, k_right])] = np.sum(
                np.abs(h_left / len(groups[k_left]) - h_right / len(groups[k_right]))) / 2
    except Exception as e:
        print(e)
        print(len(bins), len(ranges), len(data_.columns), data_.columns)
        if retargs:
            return 1, (bins, ranges)
        return 1
    if len(L1) == 1:
        if retargs:
            return list(L1.values())[0], (bins, ranges)
        return list(L1.values())[0]
    if retargs:
        return L1, (bins, ranges)
    return L1


def get_imbalance_params(data, measure, **kwargs):
    if measure == 'l1':
        imb_params = _bins_ranges_for_L1(data, kwargs.get('continuous', []), kwargs.get('H', 5))
    else:
        imb_params = {}
    return imb_params


def _bins_ranges_for_L1(data, continuous, H):
    bins = [min(x.nunique(), H) if name in continuous else x.nunique()
            for name, x in data.items()]
    ranges = [(x.min(), x.max()) for _, x in data.items()]
    return {'bins': bins, 'ranges': ranges}


def LSATT(data, treatment, outcome, weights):
    # only currently valid for dichotamous treatments

    df2 = pd.concat((data, weights.rename('weights')), axis=1)
    df2 = df2.loc[df2['weights'] > 0, :]
    res = OrderedDict()
    for i, g in df2.groupby(treatment):
        weight = g['weights'].sum()

        WSOUT = (g[outcome] * g['weights']).sum()
        wave = WSOUT / weight

        ave = g[outcome].mean()
        WSSR = (g['weights'] * (g[outcome] - ave)**2).sum()
        wstd = np.sqrt(WSSR / weight)

        res[i] = [wave, wstd, len(g)]

    return res, ttest_ind_from_stats(*list(reduce(lambda x, y: x + y, res.values())), equal_var=False)


MEASURES = {
    'l1': _L1,
}
