def fill_temp(group):
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


def clean(data):
    data = data.replace('', np.nan)
    data['temperature'] = data['temperature'].astype(float)
    data['wind'] = data['wind'].astype(int)
    data['age'] = data['age'].astype(int)
    data = data.groupby('stadium').apply(fill_temp)
    data['temperature'] = data['temperature'].fillna(70).astype(int)  # some stadiums still dont have temps.
    data['form'] = data.groupby('fkicker')['good'].transform(
        lambda row: row.ewm(span=10).mean().shift(1))  # calculate form
    data = data.drop(['home_team', 'stadium', 'fkicker'], axis=1).dropna()
    return data


def get_data(conn, date_condition, where='', xp=False, base='base_query'):
    query = open(f'../sql/{base}.sql', 'r').read()
    if not xp:
        query += '''\nwhere fg.fgxp='FG' -- not an xp'''

    query += f'''\n{where}\nand p.blk != 1 -- blocked kicks are completely unpredictable and should not be counted\
    \nand g.seas {date_condition}\
    \nand k.seas >= 0 -- some have negative seasons for some reason\
    \norder by p.pid'''

    df = pd.read_sql(query, conn, index_col='pid')

    return df
