import pandas as pd
import mysql.connector
import numpy as np

cnx = mysql.connector.connect(user='root', password='mOntie20!mysql', host='127.0.0.1', database='nfl')
sql = '''select * from player where pos1 like "%qb%"'''
df = pd.read_sql(sql, cnx)

# all empty strings
df = df.apply(lambda x: x.replace('', np.nan))

# physical
df.loc[df['hand'] < 1, 'hand'] = np.nan
df['hand'] = df.groupby('player')['hand'].apply(lambda x: x.fillna(method='ffill'))
df.loc[df['height'] < 1, 'height'] = np.nan
df.loc[df['weight'] < 1, 'weight'] = np.nan
df.loc[df['arm'] < 1, 'arm'] = np.nan

# combine
df.loc[df['broad'] < 1, 'broad'] = np.nan
df.loc[df['cone'] < 1, 'cone'] = np.nan
df.loc[df['shuttle'] < 1, 'shuttle'] = np.nan

# wonderilic and ball speed
df['wonderlic'] = np.nan
df['ball_speed'] = np.nan

cols = ['fname', 'lname', 'hand', 'arm', 'broad', 'shuttle', 'cone', 'wonderlic', 'ball_speed']
qb = df.groupby('player').first().loc[:, cols]

# lets save this as the qb combine date we have before scraping
qb.to_csv(f'../data/combine_qb_initial.csv')
