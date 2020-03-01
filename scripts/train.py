import pymc3 as pm
import pandas as pd
import mysql.connector
import pickle
from datetime import datetime as dt
import sys
sys.path.append('..')
from util.data import get_data, clean
from util.feature import calc_interactions


def get_priors(results):
    priors = {}
    for cov, data in exp_results.iterrows():
        if data['std err'] == 0:
            std = data['coef'] if data['coef'] else 0.1
        else:
            std = data['std err'] * 2
        priors[cov] = pm.Normal.dist(data['coef'], std)

    return priors


def feature_engineering(data, res):
    inters = list(filter(lambda x: '*' in x), res)
    data = calc_interactions(data, inters)
    return data


def train(data, priors, samples=1000, tune=1000):
    with pm.Model() as logistic_model:

        print('Data..', end='')
        df_x = data.drop('good', axis=1)
        shared_x = pm.Data('data', df_x)
        shared_y = pm.Data('good', data.loc[:, 'good'])

        print('! GLM..', end='')
        pm.glm.GLM(shared_x, shared_y, labels=df_x.columns.values, priors=priors, family='binomial')
        print('!')

        logistic_model.check_test_point()
        for RV in logistic_model.basic_RVs:
            print(RV.name, RV.logp(logistic_model.test_point))
        print(logistic_model.logp(logistic_model.test_point))

        # trace
        print('Start sampling.')
        trace = pm.sample(samples, tune=tune, init='adapt_diag', cores=1)
        print('Finished sampling.')

        # save model
        datestring = dt.now().strftime('%y%m%d')
        with open(f'../models/logistic_model_{datestring}.pkl', 'wb') as f:
            pickle.dump(logistic_model, f)

        # save trace
        with open(f'../models/trace_{datestring}.pkl', 'wb') as f:
            pickle.dump(trace, f)

        return logistic_model, trace


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='PyMC3 training script.')
    parser.add_argument('--data', help='Where to find pre-saved data.', default=None)
    parser.add_argument(
        '--base', help='base query to use in sql. Usually base_query or six_cat.', default=None)
    parser.add_argument('--xp', help='Include extra points in sql',
                        default=False, type=bool)
    parser.add_argument('--date', help='Dates for the sql.', default=None)
    parser.add_argument('--prior', help='Where in results/ to get the info for priors.')
    args = parser.parse_args()
    print(f'{args.xp} XP for data {args.date} with {args.base} sql and {args.prior} prior.')

    # get training data
    if args.data is not None:
        df = pd.read_csv(args.data, index_col=0)
    else:
        cnx = mysql.connector.connect(user='root', password='mOntie20!mysql',
                                      host='127.0.0.1', database='nfl')
        where = '''and (
                    (
                        fg.fkicker in (select fkicker from fifty) -- has had at least 50 attempts overall (this keeps only kickers that would end up making it in the NFL)
                    ) or    
                    (
                        k.seas>=3  -- or they had played 3 seasons up to the kick (stops unnecessary removal of kicks early or late in the dataset)
                    )
                    )'''
        df = get_data(cnx, args.date, where=where, xp=args.xp, base=args.base)

    # get priors (using results pre-2011)
    exp_results = pd.read_csv(f'../results/{args.prior}.csv', index_col=0).rename(
        index=lambda x: x.replace(':', '*'))

    # get interactions
    df = clean(df)
    df = feature_engineering(df, exp_results.index)

    # get priors for features
    priors = get_priors(exp_results)

    formula = 'good ~ ' + '+'.join(df.drop('good', axis=1).columns.values)
    print(formula)

    # params
    print(f'Training examples: {len(df)}')

    # print(df.info())
    print(df.head())
    print(list(priors.keys()))

    # train
    train(df, priors)
