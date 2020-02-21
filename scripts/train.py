import sys
sys.path.append('C:\\Users\\Lewis.Bails\\Repositories\\nfl_study')
from util.stats import summary
import pymc3 as pm
import pandas as pd
import mysql.connector
import pickle
from datetime import datetime as dt


def get_data(conn):
    base_query = '''select
    p.pid,fg.good,fg.dist, 
    g.seas as year, k.seas as seasons,
    case when g.temp<50 then 1 else 0 end as cold,
    case when g.stad like "%Mile High%" then 1 else 0 end as altitude,
    case when g.humd>=60 then 1 else 0 end as humid,
    case when g.wspd>=10 then 1 else 0 end as windy,
    case when g.v=p.off then 1 else 0 end as away_game,
    case when g.wk>=10 then 1 else 0 end as postseason,
    case when (pp.qtr=p.qtr) and ((pp.timd-p.timd)>0 or (pp.timo-p.timo)>0) then 1 else 0 end as iced,
    case g.surf when 'Grass' then 0 else 1 end as turf,
    case when g.cond like "%Snow%" then 1 when g.cond like "%Rain%" and not "Chance Rain" then 1 else 0 end as precipitation,
    case when p.qtr=4 and ABS(p.ptso - p.ptsd)>21 then 0
    when p.qtr=4 and p.min<2 and ABS(p.ptso - p.ptsd)>8 then 0
    when p.qtr=4 and p.min<2 and p.ptso-p.ptsd < -7 then 0
    when p.qtr<=3 then 0
    when p.qtr=4 and p.min>=2 and ABS(p.ptso - p.ptsd)<21 then 0
    when p.qtr=4 and p.min<2 and p.ptso-p.ptsd >=5 and p.ptso-p.ptsd <=8 then 0
    when p.qtr=4 and p.min<2 and p.ptso-p.ptsd >=-4 and p.ptso-p.ptsd <=-6 then 0
    else 1 end as pressure'''

    query = base_query + '''
    from FGXP fg
    left join PLAY p on fg.pid=p.pid
    left join game g on p.gid=g.gid
    join kicker k on k.player = fg.fkicker and g.gid=k.gid
    join PLAY pp on pp.pid=p.pid-1 and pp.gid=p.gid
    where fg.fgxp='FG' -- not an xp
    and g.seas >2017
    order by p.pid
    '''

    df = pd.read_sql(query, conn, index_col='pid')
    df['cold*windy'] = df['cold'] * df['windy']
    df['postseason*away_game'] = df['postseason'] * df['away_game']

    return df


def get_priors(results):
    priors = {}
    for cov, data in exp_results.iterrows():
        priors[cov] = pm.Normal.dist(data['coef'], data['std err'] * 2)

    return priors


def train(data, priors, samples=200, tune=500):
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


if __name__ == '__main__':
    # get training data
    cnx = mysql.connector.connect(user='root', password='mOntie20!mysql', host='127.0.0.1', database='nfl')
    df = get_data(cnx)

    # get priors (using results pre-2017)
    exp_results = pd.read_csv('../results/expanded_results.csv', index_col=0).rename(
        index={'cold:windy': 'cold*windy', 'postseason:away_game': 'postseason*away_game'})
    priors = get_priors(exp_results)

    # formula = 'good ~ ' + '+'.join(df.drop('good', axis=1).columns.values)
    # print(formula)

    # params
    print(f'Training examples: {len(df)}')

    # train
    train(df, priors)
