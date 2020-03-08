import time
from selenium import webdriver
import re
import numpy as np

temporal = r'(\d+\.\d+)\sseconds'
spatial = r'(\d+\.\d+)\sinches'
wonderlic = r'(\d+)\s\(0\-50\)'
speed = r'([\d\.]+)\s\(MPH\)'

X_MAP = {
    'hand': {'x_code': '//*[@id="player_info_wrap"]/div[3]/div/table/tbody/tr[5]/td[2]',
             'regex': spatial},
    'arm': {'x_code': '//*[@id="player_info_wrap"]/div[3]/div/table/tbody/tr[4]/td[2]',
            'regex': spatial},
    'shuttle': {'x_code': '//*[@id="measureables2"]/table/tbody/tr[3]/td[2]',
                'regex': temporal},
    'cone': {'x_code': '//*[@id="measureables2"]/table/tbody/tr[4]/td[2]',
             'regex': temporal},
    'broad': {'x_code': '//*[@id="measureables2"]/table/tbody/tr[2]/td[2]',
              'regex': spatial},
    'wonderlic': {'x_code': '//*[@id="measureables1"]/table/tbody/tr[6]/td[2]',
                  'regex': wonderlic},
    'ball_speed': {'x_code': '//*[@id="measureables1"]/table/tbody/tr[7]/td[2]',
                   'regex': speed},
}


class QBBot:

    def __init__(self):
        self.driver = webdriver.Chrome()
        self.first_name = None
        self.last_name = None

    def missing(self, attr):
        print(f'Missing {attr} for {self.first_name} {self.last_name}')

    def get_attribute(self, attr):
        pos = self.driver.find_element_by_xpath(
            '//*[@id="player_info_wrap"]/div[2]/table/tbody/tr[4]/td[2]').text
        if pos.lower() != 'quarterback':
            print(f'{self.first_name} {self.last_name} not quarterback')
            return np.nan

        time.sleep(1)
        attr_map = X_MAP.get(attr)
        if attr_map is not None:
            try:
                box = self.driver.find_element_by_xpath(attr_map['x_code'])
            except:
                self.missing(attr)
                return np.nan
            match = re.search(attr_map['regex'], box.text)
            if match is not None:
                val = float(match.group(1))
            else:
                self.missing(attr)
                return np.nan
            print(attr, self.first_name, self.last_name, val)
            return val
        else:
            self.missing(attr)
            return np.nan

    def get_player(self, first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name
        self.driver.get('https://nflcombineresults.com/psearch.php')
        time.sleep(2)
        search_btn = self.driver.find_element_by_xpath('//*[@id="search_label_top"]')
        search_btn.click()
        time.sleep(1)
        text_box = self.driver.find_element_by_xpath('//*[@id="search-text-top"]')
        text_box.send_keys(first_name + ' ' + last_name)
        text_box.send_keys(webdriver.common.keys.Keys.RETURN)
        time.sleep(2)
        try:
            list_item = self.driver.find_element_by_xpath('//*[@id="searchtable"]/tbody/tr/td[1]/a')
            list_item.click()
        except:
            print(f'Cant find {self.first_name} {self.last_name}.')
            return False
        time.sleep(1)
        return True


if __name__ == '__main__':
    import pandas as pd
    from datetime import datetime as dt
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--date')
    args = parser.parse_args()

    missing = pd.read_csv(f'../data/combine_qb_{args.date}.csv', index_col=0)

    bot = QBBot()

    for i, x in missing.iterrows():
        success = bot.get_player(x['fname'], x['lname'])
        if success:
            for j, val in x.drop(index=['fname', 'lname']).iteritems():
                if np.isnan(val):
                    missing.loc[i, j] = bot.get_attribute(j)

    today = dt.now().strftime('%y%m%d')
    missing.to_csv(f'../data/combine_qb_{today}.csv')
