from typing import Tuple
import random

import requests
from lxml import etree 
from bs4 import BeautifulSoup

def get_word_definition(word:str)->Tuple[bool, str]:
    url = f'https://www.merriam-webster.com/dictionary/{word}'

    # Make the request to the website
    response = requests.get(url, headers={'Content-Type': 'application/json'})

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        dom = etree.HTML(str(soup)) 
        definition_tag = dom.xpath('/html/head/meta[5]')[0].attrib

        if definition_tag.get('content'):
            return True, definition_tag['content']
        else:
            return False, f"No definition found for '{word}'."
    else:
        return False, f"Error {response.status_code}: Unable to fetch definition for '{word}'."
    
def get_random_word_of_the_day():
    url = f'https://www.merriam-webster.com/word-of-the-day/2022-{str(random.randint(1,12)).rjust(2,"0")}-{str(random.randint(1,26)).rjust(2,"0")}'

    # Make the request to the website
    response = requests.get(url, headers={'Content-Type': 'application/json'})

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        dom = etree.HTML(str(soup)) 
        definition_tag = dom.xpath('/html/body/div[1]/div/div[2]/main/article/div[1]/div[3]/div[1]/div/h2')[0].text

        if definition_tag:
            return True, definition_tag
        else:
            return False, f"No word found."
    else:
        return False, f"Error: Unable to fetch word."

if __name__ == "__main__":
    import time
    # words generated from the AI model
    words = ['oikofugic', 'embolalia', 'hypohippus', 'vibrissae', 'zwinger', 'extispicy', 'ozostomia', 'ribzuba', 'vumbum', 'zumbooruk', 'zobo', 'cisvestism', 'piboogism', 'didleroncie', 'ozostomia', 'liripoop', 'hackmatack', 'jotabaasia', 'liripooe']

    for word in words:
        is_word, definition = get_word_definition(word)
        print(f'{word} : {is_word} : {definition}')
        print()
        time.sleep(1)