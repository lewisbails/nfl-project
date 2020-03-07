import time
from selenium import webdriver
import re
import numpy as np


class QBBot:

    def __init__(self):
        self.driver = webdriver.Chrome()

    def search(self, first_name, last_name):
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
            return np.nan
        time.sleep(1)
        try:
            hand_box = self.driver.find_element_by_xpath(
                '//*[@id="player_info_wrap"]/div[3]/div/table/tbody/tr[5]/td[2]')
        except:
            return np.nan
        match = re.search(r'(\d+\.\d+)\sinches', hand_box.text)
        if match is not None:
            hand_size = float(match.group(1))
        else:
            return np.nan

        print(first_name, last_name, hand_size)
        return hand_size
