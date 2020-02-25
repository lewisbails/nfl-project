import pymc3 as pm
import pandas as pd
import mysql.connector
import pickle
from datetime import datetime as dt


def get_data(conn, date_condition, where='', xp=False, base='base_query'):
    query = open(f'../sql/{base}.sql', 'r').read()
    if not xp:
        query += '''\nwhere fg.fgxp='FG' -- not an xp'''

    query += f'''\n{where}\nand p.blk != 1 -- blocked kicks are completely unpredictable and should not be counted\
    \nand g.seas {date_condition}\
    \nand k.seas >= 0 -- some have negative seasons for some reason\
    \norder by p.pid'''

    # the k.seas>=3 is not necessary if the date_condition is later in the dataset, although it shouldnt affect the data too much,
    # as most kickers past 3 seasons have had 50 or more kicks.

    df = pd.read_sql(query, conn, index_col='pid')

    return df


def get_priors(results):
    priors = {}
    for cov, data in exp_results.iterrows():
        priors[cov] = pm.Normal.dist(data['coef'], data['std err'] * 2)

    return priors


def train(data, priors, samples=1000, tune=1000):
    with pm.Model() as logistic_model:

        # from_formula was working so i went with the standard GLM API
        # it didnt like the whole passing a SharedVariable in the data slot
        # if 'Intercept' not in data.columns:
        #     data['Intercept'] = 1
        # df_shared = pm.Data('data', data)
        # pm.glm.GLM.from_formula(formula,
        #                         df_shared,
        #                         priors=priors,
        #                         family='binomial')

        print('Data..', end='')
        df_x = data.drop('good', axis=1)
        shared_x = pm.Data('data', df_x)
        shared_y = pm.Data('good', data.loc[:, 'good'])

        print('! GLM..', end='')
        pm.glm.GLM(shared_x, shared_y, labels=df_x.columns.values, priors=priors, family='binomial')
        print('!')

        # trace
        print('Start sampling.')
        trace = pm.sample(samples, tune=tune, init='adapt_diag')
        print('Finished sampling.')

        # save model
        datestring = dt.now().strftime('%y%m%d')
        with open(f'../models/logistic_model_{datestring}.pkl', 'wb') as f:
            pickle.dump(logistic_model, f)

        # save trace
        with open(f'../models/trace_{datestring}.pkl', 'wb') as f:
            pickle.dump(trace, f)

        return logistic_model, trace


def get_interactions(data, res):
    for i in res.index:
        if '*' in i:
            l, r = i.split('*')
            data[i] = data[l] * data[r]

    return data


def feature_engineering(data, res):

    # get interaction features
    data = get_interactions(data, res)

    # get form feature
    data['form'] = data.groupby('fkicker')['good'].transform(lambda row: row.ewm(span=5).mean().shift(1))
    data = data.drop('fkicker', axis=1)

    return data


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='PyMC3 training script.')
    parser.add_argument('--base', help='base query to use in sql. Usually base_query or six_cat.')
    parser.add_argument('--xp', help='Include extra points in sql',
                        default=False, type=bool)
    parser.add_argument('--date', help='Dates for the sql.')
    parser.add_argument('--prior', help='Where in results/ to get the info for priors.')
    args = parser.parse_args()
    print(f'{args.xp} XP for data {args.date} with {args.base} sql and {args.prior} prior.')

    # get training data
    cnx = mysql.connector.connect(user='root', password='mOntie20!mysql', host='127.0.0.1', database='nfl')
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

    # get interactions and form features
    df = feature_engineering(df, exp_results)

    # get priors for features
    priors = get_priors(exp_results)

    formula = 'good ~ ' + '+'.join(df.drop('good', axis=1).columns.values)
    print(formula)

    # params
    print(f'Training examples: {len(df)}')

    # print(df.info())
    # print(priors)
    # import sys
    # sys.exit(1)

    # train
    train(df, priors)
