import pandas as pd
import itertools
from sklearn.preprocessing import StandardScaler


class Features:
    ''' Feature engineer '''

    def __init__(self, data, y):
        self.data = data
        self.y = y
        self.scaler = None

    def normalise(self):
        self.scaler = StandardScaler()
        normed = pd.DataFrame.from_records(self.scaler.fit_transform(self.data),
                                           columns=self.data.columns, index=self.data.index)
        normed[self.y] = self.data[self.y]
        self.data = normed
        return self

    def get_interactions(self, cols, doubles=True, triples=False):
        interactions = get_interactions(cols, doubles, triples)
        self.data = calc_interactions(self.data, interactions)
        return self

    def get_polynomials(self, cols, exp=2):
        for col in cols:
            self.data[f'{col}^{exp}'] = self.data[col]**exp
        return self


class LASSO:
    ''' Feature selection by bootstrapped penalised regression.
        Include paired or triple interactions.
    '''

    def __init__(self, data, y):
        self.features = Features(data, y)

    def fit(self, n=1, filter_=True):
        summary = self.bootstrap(self.features.data, n=n)
        if filter_:
            summary = summary[((summary['25%'] < 0) & (summary['75%'] < 0)) |
                              ((summary['25%'] > 0) & (summary['75%'] > 0))]
            summary['abs_mean'] = summary['mean'].abs()
        summary = summary.sort_values('abs_mean', ascending=False)
        print('Finished!')
        return summary

    def bootstrap(self, data, y='good', n=20):
        ''' Logisitic regression bootstrap for variable coefficients. '''
        from sklearn.linear_model import LogisticRegression
        coefs = {}
        for i in range(n):  # bootstrap for CI on coefficients
            boot = data.sample(frac=1, replace=True)
            model = LogisticRegression(penalty='l1', max_iter=1000, solver='liblinear', fit_intercept=True)
            model.fit(X=boot.drop(y, axis=1), y=boot[y])
            coefs[i] = pd.Series(index=boot.drop(y, axis=1).columns.values, data=model.coef_[0])

        return pd.DataFrame.from_dict(coefs, orient='index').describe().transpose()


def get_interactions(covariates, doubles=True, triples=False):
    ''' A list of string interactions, paired or triples, for the covariates given. '''
    interactions = []
    if covariates is not None:
        if doubles:
            for i, j in itertools.combinations(covariates, 2):
                interactions.append(i + '*' + j)

        if triples:
            for i, j, k in itertools.combinations(covariates, 3):
                interactions.append(i + '*' + j + '*' + k)

    return interactions


def calc_interactions(data, interactions):
    ''' Calculate the interaction value given the original data and interactions. '''
    for interaction in interactions:
        covs = interaction.split('*')
        if len(covs) == 2:
            data[interaction] = data[covs[0]] * data[covs[1]]
        else:
            data[interaction] = data[covs[0]] * data[covs[1]] * data[covs[2]]
    return data
