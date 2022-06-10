import requests
from bs4 import BeautifulSoup
import random

def get_soup(url=None, headers=None):
    """
    get the message form url
    :param url:
    :return:
    """
    # get the html
    data = requests.get(url=url, headers=headers)
    # get the soup class
    soup = BeautifulSoup(data.text, 'html.parser')
    # get the class
    wrap = soup.find_all(class_="content__title")
    return wrap
    pass





if __name__ == '__main__':
    print(int('93.00'))