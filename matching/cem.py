# Copyright (c) 2020 Lewis Bails
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''Coarsened exact matching for causal inference'''

from __future__ import absolute_import

import numpy as np
import pandas as pd
from pandas import Series
from pandas.core.frame import _from_nested_dict

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf
import statsmodels.api as sm

from scipy.stats import ttest_ind_from_stats
from scipy.stats import ttest_ind
from scipy.stats import chisquare

from itertools import combinations, product
from collections import OrderedDict
from functools import reduce
from copy import deepcopy
from tqdm import tqdm
from typing import Union

from util.evaluation import odds

__author__ = "Lewis Bails <lewis.bails@gmail.com>"
__version__ = "0.1.0"


class CEM:
    '''Coarsened Exact Matching

    Parameters
    ----------
    data : DataFrame
    treatment : str
        The treatment variable in data
    outcome : str
        The outcome variable in data
    continuous : list
        The continuous variables in data
    H : int, optional
        The number of bins to use for the continuous variables when calculating imbalance
    measure : str, optional
        Multivariate imbalance measure to use

    Attributes
    ---------
    data, treatment, outcome, continuous, H: see Parameters
    bins : array_like
        Array of bin edges
    preimbalance : float
        The imbalance of the data prior to matching
    measure : str
        Multivariate imbalance measure

    '''

    def __init__(self, data: pd.DataFrame, treatment: str, outcome: str, continuous: list = [], H: int = None, measure: str = 'l1'):
        self.data = data.copy()
        self.treatment = treatment
        self.outcome = outcome
        self.continuous = continuous
        self.H = H
        self.measure = measure
        self.bins = None
        self.preimbalance = None

        #  find H, get bin edges
        df = self.data.drop(outcome, axis=1)
        if self.H is None:
            print('Calculating H, this may take a few minutes.')
            rows = []
            cont_bins = range(1, 10)
            imb = []
            for h in cont_bins:
                bins = get_imbalance_params(df.drop(self.treatment, axis=1),
                                            self.measure, self.continuous, h)
                l1 = imbalance(df, self.treatment, self.measure, bins)
                imb.append(l1)
            imb = pd.Series(imb, index=cont_bins)
            self.H = (imb.sort_values(ascending=False) <= imb.quantile(.5)).idxmax()

        self.bins = get_imbalance_params(df.drop(self.treatment, axis=1),
                                         self.measure, self.continuous, self.H)
        self.preimbalance = imbalance(df, self.treatment, self.measure, self.bins)

    def imbalance(self, coarsening: dict, one_to_many: bool = True) -> float:
        '''Calculate the imbalance remaining after matching the data using some coarsening

        Parameters
        ----------
        coarsening : dict
            Defines the strata.
            Keys are the covariate names and values are dict's themselves with keys of "bins" and "cut"
            "bins" is the first parameter to the "cut" method stipulated (i.e. number of bins or bin edges, etc.)
            "cut" is the Pandas method to use for grouping the covariate (only "cut" and "qcut" supported)
        one_to_many : bool
            Whether to limit the matches in a stratum to k:k, or allow k:n matches.

        Returns
        -------
        float
            The residual imbalance
        '''

        weights = self.match(coarsening, one_to_many)
        df = self.data.drop(self.outcome, axis=1).loc[weights > 0, :]
        return imbalance(df, self.treatment, self.measure, self.bins)

    def univariate_imbalance(self, coarsening: dict, one_to_many: bool = True):
        '''Calculate the marginal imbalance remaining for each covariate post-matching

        Parameters
        ----------
        coarsening, one_to_many: See imbalance.

        Returns
        -------
        UnivariateImbalance:
            Covariate residual imbalance and a plotting option.

        '''
        weights = self.match(coarsening, one_to_many)
        df = self.data.drop(self.outcome, axis=1).loc[weights > 0, :]
        return UnivariateBalance(df, self.treatment, self.measure, self.bins)

    def match(self, coarsening: dict, one_to_many: bool = True) -> pd.Series:
        ''' Perform CEM using some coarsening schema

        Parameters
        ----------
        coarsening, one_to_many: See imbalance.

        Returns
        -------
        Series
            Weights for each data point
        '''
        return match(self.data.drop(self.outcome, axis=1), self.treatment, coarsening, one_to_many)

    def relax(self, coarsening: dict, relax_vars: 'array_like'):
        ''' Evaluate the residual imbalance and match information for several coarsenings.

        Parameters
        ----------
        coarsening: See imbalance.
        relax_vars: array_like
            3-tuples, combined to progressively modify the coarsening.
            3-tuple is (name, iterable, cut_method).
            The iterables' cartesian product is used for producing the combinations.

        Returns
        -------
        Relax
            Progressive coarsening results and plotting option
        '''
        return Relax(self.data.drop(self.outcome, axis=1), self.treatment, coarsening, relax_vars, self.measure,
                     bins=self.bins)

    def regress(self, coarsening: dict,
                relax_vars: 'array_like' = [],
                formula: str = None,
                drop: 'array_like' = [],
                include_relax: bool = True,
                family=None):
        ''' Perform a log-linear regression for one or more coarsenings

        Parameters
        ----------
        coarsening, relax_vars: See relax
        formula: str, optional
            GLM-like formula for the statsmodels model instance.
        drop: array_like, optional
            Not required if formula is given. Denotes covariates to drop from regression.
        include_relax: bool, optional
            Whether to return Relax which containing matching information for each coarsening.

        Returns
        -------
        Regress
            Regression results and plotting option
        Relax
            Progressive coarsening results and plotting option
        '''
        reg = Regress(self.data, self.treatment, self.outcome, coarsening,
                      relax_vars=relax_vars, formula=formula,
                      drop=drop, family=family)
        if include_relax:
            rel = self.relax(coarsening, relax_vars)
            return reg, rel
        return reg

    def LATE(self, coarsening: dict, one_to_many: bool = True) -> tuple:
        '''Local average treatment effect for some coarsening'''
        weights = self.match(coarsening, one_to_many)
        return LATE(self.data, self.treatment, self.outcome, weights)


class UnivariateBalance:
    '''Residual marginal imbalance for each covariate post-matching

    Parameters
    ----------
    data: DataFrame
        On which to calculate imbalance.
    treatment: str
        The variable defining treatment groups.
    measure: str
        Univariate imbalance measure to use.
    bins: array_like, optional
        Bin edges for constructing the histograms

    Attributes
    ----------
    data, treatment, measure, bins: See parameters.
    summary: pd.DataFrame
        Imbalance statistics for each covariate
    '''

    def __init__(self, data: pd.DataFrame, treatment: str, measure: str, bins: 'array_like', weights=None):
        assert len(data.drop(treatment, axis=1).columns) == len(bins), 'Lengths not equal.'
        if measure not in ('l1', 'l2'):
            raise NotImplementedError('Only L1/2 possible at the moment.')
        self.data = data
        self.treatment = treatment
        self.measure = measure
        self.bins = bins
        self.weights = weights
        self.summary = self._summarise(self.data)

    # def _summarise(self) -> pd.DataFrame:
    #     '''Calculate the marginal imbalances and return summary statistics'''
    #     if self.weights:
    #         return pd.concat({i: self._summarise_one(self.data.loc[w > 0, :]) for i, w in self.weights.items()})
    #     else:
    #         return self._summarise_one(self.data)

    def _summarise(self, data):
        marginal = {}
        # it is assumed the elements of bins lines up with the data (minus the treatment column)
        for col, bin_ in zip(data.drop(self.treatment, axis=1).columns, self.bins):
            cem_imbalance = imbalance(data.loc[:, [col, self.treatment]],
                                      self.treatment, self.measure, [bin_])
            d_treatment = data.loc[data[self.treatment] > 0, col]
            d_control = data.loc[data[self.treatment] == 0, col]
            if data[col].nunique() > 2:
                stat = d_treatment.mean() - d_control.mean()
                _, p = ttest_ind(d_treatment, d_control, equal_var=False)
                type_ = 'diff'
            else:  # binary variables
                f_obs = d_treatment.value_counts(normalize=True)
                f_exp = d_control.value_counts(normalize=True)
                stat, p = chisquare(f_obs, f_exp)
                type_ = 'Chi2'

            q = [0, 0.25, 0.5, 0.75, 1]
            diffs = d_treatment.quantile(
                q) - d_control.quantile(q) if type_ == 'diff' else pd.Series([None] * len(q), index=q)
            row = {'imbalance': cem_imbalance, 'measure': self.measure,
                   'statistic': stat, 'type': type_, 'P>|z|': p}
            row.update({f'{int(i*100)}%': diffs[i] for i in q})
            marginal[col] = pd.Series(row)
        return pd.DataFrame.from_dict(marginal, orient='index')

    def hist(self, kde=False, hist=True):
        '''Marginal histogram/density plots'''
        vals = self.data[self.treatment].unique()
        flatui = ["#2ecc71", "#9b59b6", "#3498db", "#e74c3c", "#34495e"]  # TODO: change this

        for col, bin_ in zip(self.data.drop(self.treatment, axis=1).columns, self.bins):
            try:
                for i, val in enumerate(vals):
                    sns.distplot(self.data[self.data[self.treatment] == val][col],
                                 bins=bin_, label=f'{self.treatment}={val}', kde=kde, norm_hist=hist, hist=hist, color=flatui[i])
                    plt.axvline(self.data[self.data[self.treatment] == val][col].mean(), color=flatui[i])
            except:
                for i, val in enumerate(vals):
                    sns.distplot(self.data[self.data[self.treatment] == val][col],
                                 bins=bin_, label=f'{self.treatment}={val}', kde=False, norm_hist=True, hist=True, color=flatui[i])
                    plt.axvline(self.data[self.data[self.treatment] == val][col].mean(), color=flatui[i])

            plt.title(f'{col} distributions')
            plt.show()


class Regress:
    '''Summarise results of a regression process on potentially several coarsenings.

    Parameters
    ----------
    data: DataFrame
        On which to calculate imbalance.
    treatment: str
        The variable defining treatment groups.
    outcome: str
        The outcome variable in data
    coarsening: dict
        Base coarsening schema
    relax_vars: array_like, optional
        3-tuples for defining progressive coarsening
    formula: str, optional
        GLM-like regression formula
    drop: array_like, optional
        Covariates to remove before regression

    '''

    def __init__(self,
                 data: pd.DataFrame,
                 treatment: str,
                 outcome: str,
                 coarsening: dict,
                 relax_vars: 'array_like' = [],
                 formula: str = None,
                 drop: 'array_like' = [],
                 family=None):
        self.data = data
        self.treatment = treatment
        self.outcome = outcome
        self.base_coarsening = coarsening
        self.relax_vars = relax_vars
        self.formula = formula
        self.drop = drop
        self.results = regress(data, treatment, outcome, coarsening, relax_vars, formula, drop, family)
        if isinstance(self.results, pd.Series):  # relax_vars was empty
            self.results['n_bins'] = None
            self.results['var'] = None
            self.results = pd.DataFrame([self.results]).set_index(['var', 'n_bins'])

    def _sm_summary_to_frame(self, summary: 'statsmodels.Summary') -> pd.DataFrame:
        '''Convert Summary object to DataFrame of covariate regression statistics'''
        pd_summary = pd.read_html(summary.tables[1].as_html())[0]
        pd_summary.iloc[0, 0] = 'covariate'
        pd_summary.columns = pd_summary.iloc[0]
        pd_summary = pd_summary[1:]
        pd_summary.set_index('covariate', inplace=True)
        return pd_summary.astype(float)

    def _row_to_long(self, row: pd.Series) -> pd.DataFrame:
        '''Convert row containing sm.Result and progressive coarsening information
            to a DataFrame of covariate regression statistics'''
        summary = self._sm_summary_to_frame(row['result'].summary())
        summary['var'] = row['var']
        summary['n_bins'] = row['n_bins']
        summary.set_index(['n_bins', 'var'], append=True, inplace=True)
        return summary

    def expand(self) -> pd.DataFrame:
        '''Expanded regression results'''
        return pd.concat([self._row_to_long(row) for _, row in self.results.reset_index().iterrows()])

    def _annotate(self, data: pd.DataFrame, ax: 'Axes') -> 'Axes':
        '''P-value annotations for the progressive coarsening plot'''
        for j, row in data.iterrows():
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

    def plot(self, include: 'array_like' = [], stars: bool = True, legend='full') -> 'Axes':
        '''Plot regression results from progressive coarsening

        Parameters
        ----------
        include: array_like
            Covariates to include in the plot
        stars: bool
            Whether to include the annotations indicating P-values
        '''

        lf = self.expand().reset_index()
        if lf['var'].nunique() > 1:
            raise Exception('Progressive coarsening plot only available for single variable.')
        else:
            var = lf['var'].iloc[0]

        fig, ax = plt.subplots()
        r = lf.reset_index()
        if len(include):
            r = r.loc[r['covariate'].isin(include), :]
        else:
            r = r.loc[~r['covariate'].isin(['Intercept']), :]

        r['P>|z|'] = r['P>|z|'].round(2)

        ax = sns.lineplot(x='n_bins', y='coef', hue='covariate', data=r, legend=False)

        ax = sns.scatterplot(x='n_bins', y='coef', hue='covariate', size='P>|z|',
                             sizes=(0, 150), data=r, ax=ax, legend=legend)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        if stars:
            ax = self._annotate(r, ax)

        fig.set_size_inches(12, 8)
        ax.set_title('Regression coefficients for progressive coarsening.')
        ax.set_ylabel('Coefficient')
        ax.set_xlabel(f'# bins for {var} coarsening')
        return ax

    def odds_ratios(self, hue, hue_values, p=0.1, plot=True):
        rows = []
        lf = self.expand().reset_index(['n_bins', 'var'])
        var = lf['var'].iloc[0]
        for i, g in lf.groupby('n_bins'):
            for v in hue_values:
                base = odds(g['coef'], g['P>|z|'] <= p, **{self.treatment: False, hue: v})
                odds_Xi = odds(g['coef'], g['P>|z|'] <= p, **{self.treatment: True, hue: v})
                OR = odds_Xi / base
                row = {'n_bins': i, 'OR': OR, f'{hue}': v}
                rows.append(row)
        res = pd.DataFrame.from_records(rows)
        if plot:
            ax = sns.catplot(x='n_bins', y='OR', hue=hue, data=res, kind='bar')
            ax.set_axis_labels(f'# bins for {var} coarsening', 'Odds Ratio')
            ax.fig.suptitle(f'Odds ratios for {self.treatment} at varying {hue} and # {var} bins')
            ax.fig.set_size_inches(12, 8)
        return res


class Relax:
    '''Summarise the results of progressive coarsening

    Parameters
    ----------
    data: DataFrame
        On which to match
    treatment: str
        Variable defining treatment groups
    coarsening: dict
        Base coarsening schema
    relax_vars: array_like
        3-tuples for defining progressive coarsening
    measure: str
        Multivariable imbalance measure
    continuous: array_like, optional
        Continuous variables in data
    **bins: array_like, optional
        Bin edges for evaluating imbalance

    Attributes
    ----------
    coarsenings: DataFrame
        Imbalance and matching statistics from each coarsening

    '''

    def __init__(self,
                 data: pd.DataFrame,
                 treatment: str,
                 coarsening: dict,
                 relax_vars: 'array_like',
                 measure: str = 'l1',
                 continuous: 'array_like' = [],
                 **kwargs):
        self.total = len(data)
        self.relax_vars = relax_vars
        self.base_coarsening = coarsening
        self.measure = measure
        self.continuous = continuous
        self.__dict__.update(**kwargs)
        self.coarsenings = relax(data, treatment, coarsening, relax_vars, measure, continuous, **kwargs)

    def _plot_multivariate(self, **kwargs):
        '''Plot the percentage of observations matched against the coarsening and annotate with imbalance'''
        s = self.coarsenings.copy()
        t_cols = [col for col in self.coarsenings.columns if 'treatment' in col]
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
        fig, ax = plt.subplots()
        ax = sns.lineplot(x=x, y='% matched', data=s, style='measure', markers=True, **kwargs)
        fig.set_size_inches(12, 8)
        ax.set_title('Multivariate L1 for progressive coarsening')
        for _, row in s.iterrows():
            ax.text(row[x] + 0.1, row['% matched'] + 0.1, round(row['imbalance'], 2), fontsize=10)
        return ax, s

    def _plot_univariate(self, **kwargs):
        s = self.coarsenings.copy()
        x = 'n_bins'
        if len(self.relax_vars) > 1:
            s = s.reset_index('n_bins')
            x = 'coarsening'
            s[x] = range(s['n_bins'].nunique())
            s = s.set_index(x, append=True)

        fig, ax = plt.subplots()
        s = pd.concat({i: j['univariate'].summary for i, j in s.iterrows()})
        s.index.set_names(['var', x, 'covariate'], inplace=True)
        s = s.reset_index()
        ax = sns.lineplot(x=x, y='imbalance', hue='covariate', data=s, ax=ax, **kwargs)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        fig.set_size_inches(12, 8)

    def plot(self, name, **kwargs):
        if name in ('multivariate', 'multi'):
            return self._plot_multivariate(**kwargs)
        elif name in ('univariate', 'uni'):
            return self._plot_univariate(**kwargs)
        else:
            raise Exception(f'Unknown plot name "{name}"')

# The user can use all the functions outside of the classes if they choose to


def match(data, treatment, bins, one_to_many=True):
    '''Return weights for data given a coursening schema'''
    # coarsen based on supplied bins
    data_ = coarsen(data.copy(), bins)

    # weight data in non-empty strata
    if one_to_many:
        return weight(data_, treatment)
    else:
        raise NotImplementedError
        # TODO: k:k matching using bhattacharya for each stratum, weight is 1 for the control and its treatment pair


def weight(data, treatment):
    '''Weight observations based on global and strata populations'''
    # only keep stata with examples from each treatment level
    gb = list(data.drop(treatment, axis=1).columns.values)
    prematched_weights = pd.Series([0] * len(data), index=data.index)
    matched = data.groupby(gb).filter(lambda x: len(
        x[treatment].unique()) == len(data[treatment].unique()))
    if not len(matched):
        return prematched_weights
    counts = matched[treatment].value_counts()
    weights = matched.groupby(gb)[treatment].transform(lambda x: _weight_stratum(x, counts))
    return weights.add(prematched_weights, fill_value=0)


def _weight_stratum(stratum, M):
    '''Calculate weights for observations in an individual stratum'''
    ms = stratum.value_counts()
    T = stratum.max()  # use as "under the policy" level
    return pd.Series([1 if c == T else (M[c] / M[T]) * (ms[T] / ms[c]) for _, c in stratum.iteritems()])


def _bins_gen(base_coarsening, relax_on):
    '''Individual coarsening schema generator'''
    name = [v[0] for v in relax_on]
    cut_types = [v[-1] for v in relax_on]
    bins = [v[1] for v in relax_on]
    combinations = product(*bins)
    for c in combinations:
        dd = deepcopy(base_coarsening)
        new = {i: {'bins': j, 'cut': k} for i, j, k in zip(name, c, cut_types)}
        dd.update(new)
        yield dd


def relax(data, treatment, coarsening, relax_vars, measure='l1', continuous=[], include_univariate=True, **kwargs):
    '''Match on several coarsenings and evaluate some imbalance measure'''
    assert all([len(x) == 3 for x in relax_vars]
               ), 'Expected variables to relax on as tuple triples (name, iterable, cut method)'
    data_ = data.copy()
    length = np.prod([len(x[1]) for x in relax_vars])

    if 'bins' not in kwargs:
        bins = get_imbalance_params(data_.drop(
            treatment, axis=1), measure, continuous)  # indep. of any coarsening
    else:
        bins = kwargs['bins']

    rows = []
    for coarsening_i in tqdm(_bins_gen(coarsening, relax_vars), total=length):
        weights = match(data_, treatment, coarsening_i)
        nbins = np.prod([x['bins'] if isinstance(x['bins'], int) else len(x['bins']) - 1
                         for x in coarsening_i.values()])
        row = {'var': tuple(i[0] for i in relax_vars) if len(relax_vars) > 1 else relax_vars[0][0],
               'n_bins': tuple(coarsening_i[i[0]]['bins'] for i in relax_vars) if len(relax_vars) > 1 else coarsening_i[relax_vars[0][0]]['bins']}
        if (weights > 0).sum():
            d = data_.loc[weights > 0, :]
            if treatment in coarsening_i:
                # continuous treatment binning
                d[treatment] = _cut(d[treatment], coarsening_i[treatment]
                                    ['cut'], coarsening_i[treatment]['bins'])
            score = imbalance(d, treatment, measure, bins)
            vc = d[treatment].value_counts()
            row.update({'imbalance': score,
                        'measure': measure,
                        'coarsening': coarsening_i,
                        'bins': nbins})
            row.update({f'treatment_{t}': c for t, c in vc.items()})
            if include_univariate:
                row.update({'univariate': UnivariateBalance(d, treatment, measure, bins)})
        else:
            row.update({'imbalance': 1,
                        'measure': measure,
                        'coarsening': coarsening_i,
                        'bins': nbins})
        rows.append(pd.Series(row))

    return pd.DataFrame.from_records(rows).set_index(['var', 'n_bins'])


def regress(data, treatment, outcome, coarsening, relax_vars=[], formula=None, drop=[], family=None):
    '''Regress on 1 or more coarsenings and return the Results for each'''
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
            row = regress(data_, treatment, outcome, coarsening_, formula=formula)  # recurse
            row['n_bins'] = i
            row['var'] = k
            rows.append(row)
        frame = pd.DataFrame.from_records(rows)
        frame.set_index(['var', 'n_bins'], inplace=True)
        return frame
    else:
        weights_ = match(data_.drop(outcome, axis=1), treatment, coarsening_)
        res = _regress_matched(data_, formula, weights_, family)
        return pd.Series({'result': res})


def _regress_matched(data, formula, weights, family):
    glm = smf.glm(formula,
                  data=data.loc[weights > 0, :],
                  family=family,
                  var_weights=weights[weights > 0])
    result = glm.fit(method='bfgs')
    return result


def _infer_formula(data, dv, drop):
    iv = ' + '.join(data.drop([dv] + drop, axis=1).columns.values)
    return f'{dv} ~ {iv}'


def _cut(col, method, bins):
    '''Group values in a column into n bins using some Pandas method'''
    if method == 'qcut':
        return pd.qcut(col, q=bins, labels=False)
    elif method == 'cut':
        return pd.cut(col, bins=bins, labels=False)
    else:
        raise Exception(
            f'"{method}" not supported. Coarsening only possible with "cut" and "qcut".')


def coarsen(data, coarsening):
    '''Coarsen data based on schema'''
    df_coarse = data.apply(lambda x: _cut(
        x, coarsening[x.name]['cut'], coarsening[x.name]['bins']) if x.name in coarsening else x, axis=0)
    return df_coarse


def imbalance(data, treatment, measure, bins):
    '''Evaluate multivariate imbalance'''
    if measure in MEASURES:
        return MEASURES[measure](data, treatment, bins)
    else:
        raise NotImplementedError(f'"{measure}" not a valid measure. Choose from {list(MEASURES.keys())}')


def _L1(data, treatment, bins):
    def func(l, r, m, n): return np.sum(np.abs(l / m - r / n)) / 2
    return _L(data, treatment, bins, func)


def _L2(data, treatment, bins):
    def func(l, r, m, n): return np.sqrt(np.sum((l / m - r / n)**2)) / 2
    return _L(data, treatment, bins, func)


def _L(data, treatment, bins, func):
    '''Evaluate Multidimensional Ln score'''
    groups = data.groupby(treatment).groups
    data_ = data.drop(treatment, axis=1).copy()

    try:
        h = {}
        for k, i in groups.items():
            h[k] = np.histogramdd(data_.loc[i, :].to_numpy(), density=False, bins=bins)[0]
        L = {}
        for pair in map(dict, combinations(h.items(), 2)):
            pair = OrderedDict(pair)
            (k_left, k_right), (h_left, h_right) = pair.keys(), pair.values()  # 2 keys 2 histograms
            L[tuple([k_left, k_right])] = func(h_left, h_right, len(groups[k_left]), len(groups[k_right]))

    except Exception as e:
        print(e)
        return 1
    if len(L) == 1:
        return list(L.values())[0]
    return L


def get_imbalance_params(data, measure, continuous=[], H=5):
    if measure == 'l1' or measure == 'l2':
        return _bins_for_L(data, continuous, H)
    else:
        raise NotImplementedError('Only params for L variants imbalance available')


def _bins_for_L(data, continuous, H):
    def nbins(n, s): return min(s.nunique(), H) if n in continuous else s.nunique()
    bin_edges = [np.histogram_bin_edges(x, bins=nbins(i, x)) for i, x in data.items()]
    return bin_edges


def LATE(data, treatment, outcome, weights):
    '''(Weighted) Local Average Treatment Effect'''
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
    'l2': _L2,
}
