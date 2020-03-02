import pandas as pd
import numpy as np


def fill_temp(group):
    group = group.sort_index()  # index is pid
    window = 24
    while group['temperature'].isna().sum() > 0:
        # print('Window of',window)
        if window >= 48 or window >= len(group):
            # print(f'Filling for {group.name}')
            # last recorded temp was 48 kicks ago, thats too long. just fill the rest with the mean
            group['temperature'] = group['temperature'].fillna(group['temperature'].mean())
            break
        # print(group['temperature'].isna().sum(), 'are NA of', len(group), f'for {group.name}')
        rolling = group['temperature'].rolling(window, min_periods=2).mean()
        # dont replace values that arent NA
        roll_fill = rolling.drop(index=group['temperature'].dropna().index).dropna()
        group.loc[roll_fill.index, 'temperature'] = roll_fill
        # print(group['temperature'].isna().sum(), 'are NA of ', len(group), f'for {group.name}\n')
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
    data = data.replace('', np.nan)
    data['temperature'] = data['temperature'].astype(float)
    data['wind'] = data['wind'].astype(int)
    data['age'] = data['age'].astype(int)
    data = data.groupby('stadium').apply(fill_temp)  # exponentiated weighted fill
    data['temperature'] = data['temperature'].fillna(70).astype(int)  # some stadiums still dont have temps.
    data['form'] = data.groupby('fkicker')['good'].transform(
        lambda row: row.ewm(span=10).mean().shift(1))  # calculate form (exponentiated weighted)
    data = data.groupby('fkicker').apply(add_kicks)  # total kicks to date
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

    df = pd.read_sql(query, conn, index_col='pid')

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
