import pandas as pd
import numpy as np


def fill_temp(group):
    if group['stadium'].iloc[0] in ('Closed Roof', 'Dome'):
        group['temperature'] = group['temperature'].fillna(70).astype(int)
    else:
        group = group.sort_index()  # index is pid
        window = 24
        while group['temperature'].isna().sum() > 0:
            if window >= 48 or window >= len(group):
                # last recorded temp was 48 kicks ago, thats too long. just fill the rest with the mean
                group['temperature'] = group['temperature'].fillna(group['temperature'].mean())
                break
            rolling = group['temperature'].rolling(window, min_periods=2).mean()
            # dont replace values that arent NA
            roll_fill = rolling.drop(index=group['temperature'].dropna().index).dropna()
            group.loc[roll_fill.index, 'temperature'] = roll_fill
            window += 8
    return group


akps = 45  # average kicks per season


def add_kicks(group):
    group = group.sort_index()  # index is pid
    f_row = group.iloc[0, :]
    start = max(0, akps * (f_row['seasons'] - (f_row['year'] - 2000)))
    kicks = list(range(start, start + len(group)))
    group['kicks'] = pd.Series(kicks, index=group.index)
    return group


def clean(data, dropna=True):
    # data types
    data['temperature'] = data['temperature'].astype(float)
    data['wind'] = data['wind'].astype(int)
    data['age'] = data['age'].astype(int)

    # temperature imputation
    data = data.groupby(['stadium', 'year']).apply(fill_temp).droplevel(
        ['stadium', 'year'])  # exponentiated weighted fill
    data['temperature'] = data['temperature'].fillna(70).astype(int)  # some stadiums still dont have temps.
    data['temperature'] = ((data['temperature'] - 32) * 5 / 9).astype(int)  # to celcius

    # form variables
    data['form'] = data.groupby('fkicker')['good'].transform(
        lambda row: row.ewm(span=10).mean().shift(1))  # calculate form (exponentiated weighted)
    data = data.groupby('fkicker').apply(add_kicks).droplevel('fkicker')  # total kicks to date
    data['form'] = data['form'].fillna(method='bfill')

    # drop na
    if dropna:
        data = data.dropna()
    return data


def get_data(conn, date_condition, xp=False, base='base_query'):
    query = open(f'../sql/{base}.sql', 'r').read()
    if not xp:
        fg = '''\nand fg.fgxp='FG' -- not an xp'''
    else:
        fg = ''

    query += f'''\nand {date_condition}\
    \nand k.seas >= 0 -- some have negative seasons for some reason\
    \n{fg}\
    \norder by p.pid'''

    df = pd.read_sql(query, conn, index_col='pid').replace('', np.nan)

    return df


def kicks_to_date(data):
    kicks_per_kicker = {}
    fkicker_kicks_at_pid = []
    for index, row in data.iterrows():
        if row['fkicker'] not in kicks_per_kicker:
            kicks_per_kicker[row['fkicker']] = 0
        kicks_per_kicker[row['fkicker']] += 1
        fkicker_kicks_at_pid.append(kicks_per_kicker['fkicker'])
    return fkicker_kicks_at_pid


if __name__ == '__main__':
    import mysql.connector
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='field_goals')
    args = parser.parse_args()

    # load and clean the data
    cnx = mysql.connector.connect(user='root', password='mOntie20!mysql', host='127.0.0.1', database='nfl')
    df = get_data(cnx, 'g.seas<=2019', xp=False, base='raw_6_cat')
    df = clean(df, dropna=False)
    df = df.drop(['fkicker', 'home_team', 'stadium', 'team', 'XP', 'humid'], axis=1)
    df.to_csv(f'../data/{args.name}.csv')
