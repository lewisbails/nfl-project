import pandas as pd
from .tests import LR_test


class Summary():

    def __init__(self, result):

        # model results
        self.params = set(result.params.index)
        self.summary = self.pd_summary(result)
        self.result = result
        self.log_likelihood = base_ll = pd.read_html(result.summary().tables[0].as_html())[0].iloc[4, 3]

    def pd_summary(self, res):
        return pd.read_html(res.summary().tables[1].as_html(), header=0, index_col=0)[0]

    def compare(self, other):
        # other is without the covariate in question
        cov = list(self.params - set(other.result.params.index))
        dof = len(cov)
        if len(cov) == 1:
            cov = cov[0]
        else:
            cov = 'multi'

        p = LR_test(other.log_likelihood, self.log_likelihood, dof)
        other_summary = other.summary

        diff = self.summary.sub(other_summary)

        diff[f'coef with_{cov}'] = self.summary['coef']
        diff[f'coef w/o_{cov}'] = other_summary['coef']
        diff[f'% coef'] = (self.summary['coef'] - other_summary['coef']) / self.summary['coef'] * 100
        diff[f'se with_{cov}'] = self.summary['std err']
        diff[f'se w/o_{cov}'] = other_summary['std err']
        diff[f'% se'] = (self.summary['std err'] - other_summary['std err']) / \
            self.summary['std err'] * 100
        diff[f'P with_{cov}'] = self.summary['P>|z|']
        diff[f'P w/o_{cov}'] = other_summary['P>|z|']
        diff.drop(['std err', 'P>|z|', 'coef', 'z', '[0.025', '0.975]'], axis=1, inplace=True)
        return diff, p

    def __str__(self):
        return str(self.result.summary())

    def __sub__(self, other):
        return self.compare(other)
