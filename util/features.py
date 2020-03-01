import pandas as pd
import itertools


def LASSO(data, covarients, y='good', doubles=True, triples=True, drop_post=[], filter=True, n=20):
    ''' Feature selection by bootstrapped penalised regression.
        Include paired or triple interactions.
    '''

    print('Normalising..', end=' ')
    normed = pd.DataFrame.from_records(StandardScaler().fit_transform(data),
                                       columns=data.columns, index=data.index)
    normed[y] = data[y]
    print('Calculating interactions..', end=' ')
    inters = get_interactions(covarients, doubles=doubles, triples=triples)
    normed = calc_interactions(normed, inters)
    normed = normed.drop(drop_post, axis=1)
    print('Fitting bootstraps..', end=' ')
    summary = bootstrap(normed, n=n)
    if filter:
        print('Filtering..', end=' ')
        summary = summary[((summary['25%'] < 0) & (summary['75%'] < 0)) |
                          ((summary['25%'] > 0) & (summary['75%'] > 0))]
    summary['abs_mean'] = summary['mean'].abs()
    summary = summary.sort_values('abs_mean', ascending=False)
    print('Finished!')
    return summary


def get_interactions(covariates, doubles=True, triples=False):
    ''' A list of string interactions, paired or triples, for the covariates given. '''
    interactions = []
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


def bootstrap(data, y='good', n=20):
    ''' Logisitic regression bootstrap for variable coefficients. '''
    from sklearn.linear_model import LogisticRegression
    coefs = {}
    for i in range(n):  # bootstrap for CI on coefficients
        boot = data.sample(frac=1, replace=True)
        model = LogisticRegression(penalty='l1', max_iter=1000, solver='liblinear', fit_intercept=True)
        model.fit(X=boot.drop(y, axis=1), y=boot[y])
        coefs[i] = pd.Series(index=boot.drop(y, axis=1).columns.values, data=model.coef_[0])

    return pd.DataFrame.from_dict(coefs, orient='index').describe().transpose()
