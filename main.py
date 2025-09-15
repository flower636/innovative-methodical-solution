import time
import unittest

import requests
from hashlib import sha256
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from datetime import datetime
from selenium import webdriver as wd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait

NASA_VOYAGER_1_URL = "https://science.nasa.gov/mission/voyager/voyager-1/"
RFC1149_HISTORY_URL = "https://datatracker.ietf.org/doc/rfc1149/history/"
UNICODE_URL = "https://unicode.org/Public/emoji/latest/emoji-test.txt"
GENESIS_BLOCK_BITCOIN_URL = "https://www.blockchain.com/explorer/blocks/btc/000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"
KR2_ISBN10_URL = "https://search.catalog.loc.gov/instances/9acb1e70-9ea7-5ec1-9e9e-4d1e8b6d865e"

st_accept = "text/html"
st_useragent = "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Mobile Safari/537.36"
headers = {
    "Accept": st_accept,
    "User-Agent": st_useragent
}





def get_html(url: str) -> BeautifulSoup:
    req = requests.get(url, headers)
    soup = BeautifulSoup(req.text, 'lxml')
    return soup


def get_voyager_date():
    voyager_date = None
    soup = get_html(NASA_VOYAGER_1_URL)
    trs = soup.find_all("tr")
    for tr in trs:
        tds = tr.find_all("td")
        good = False
        for td in tds:
            if "launch" in td.text.lower() and "date" in td.text.lower():
                good = True
        if good:
            voyager_date = tds[1].text.split(" / ")[0].strip().split()
            voyager_date[0] = voyager_date[0][:3]
            voyager_date = ' '.join(voyager_date)
            voyager_date = datetime.strftime(datetime.strptime(voyager_date, "%b %-d, %Y"), "%Y%m%d")
    print(voyager_date)
    return voyager_date


def get_rfc1149_date():
    rfc1149_date = None
    soup = get_html(RFC1149_HISTORY_URL)
    trs = soup.find_all("tr")
    for tr in trs:
        if 'published' in tr.text.lower():
            rfc1149_date = tr.find_all("td")[0].text.strip()
            rfc1149_date = datetime.strftime(datetime.strptime(rfc1149_date, "%Y-%m-%d"), "%Y%m%d")
    print(rfc1149_date)
    return rfc1149_date


def get_brain_codepoint():
    brain_codepoint = None
    emojis = get_html(UNICODE_URL).text.split('\n')
    for emoji in emojis:
        if "brain" in emoji:
            brain_codepoint = emoji.split()[0]
    print(brain_codepoint)
    return brain_codepoint


def get_btc_genesis_date():
    browser = wd.Chrome()
    # browser.set_page_load_timeout(30)
    browser.get(GENESIS_BLOCK_BITCOIN_URL)
    time.sleep(3)
    # WebDriverWait(browser, 30).until(EC.visibility_of_element_located((By.CLASS_NAME, "sc-c317e547-9 gfpxLK")))
    soup = BeautifulSoup(browser.page_source, "lxml")
    browser.close()
    divs = soup.find("span", class_="sc-c317e547-9 gfpxLK").text
    btc_genesis_date = divs.split(', ')[0]
    btc_genesis_date = datetime.strftime(datetime.strptime(btc_genesis_date, "%d.%-m.%Y"), "%Y%m%d")
    print(btc_genesis_date)
    return btc_genesis_date


def get_kr2_isbn10():
    kr2_isbn10 = None
    browser = wd.Chrome()
    # browser.set_page_load_timeout(30)
    browser.get(KR2_ISBN10_URL)
    # time.sleep(15)
    WebDriverWait(browser, 15).until(EC.presence_of_element_located((By.ID, "react-tabs-2")))
    mac_button = browser.find_element(By.ID, "react-tabs-2")
    mac_button.click()
    time.sleep(2)
    soup = BeautifulSoup(browser.page_source, "lxml")
    browser.close()
    trs = soup.find_all("tr")
    for tr in trs:
        tds = tr.find_all("td")
        if not tds: continue
        if '020' in tds[0].text and len(tds[1].text.split()[1]) == 10:
            kr2_isbn10 = tds[1].text.split()[1]
    print(kr2_isbn10)
    return kr2_isbn10


def main():
    answer = f"FLAG{{{get_voyager_date()}-{get_rfc1149_date()}-{get_brain_codepoint()}-{get_btc_genesis_date()}-{get_kr2_isbn10()}}}"
    print(answer)
    return answer


if __name__ == "__main__":
    main()
