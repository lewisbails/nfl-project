if __name__ == '__main__':

    import argparse
    import pandas as pd
    import mysql.connector
    from train import get_data

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
    from datetime import datetime as dt
    now = dt.now().strftime('%d%m%y')
    df.to_csv(f'{now}.csv')
