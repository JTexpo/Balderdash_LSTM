from typing import List
import csv
import random

from pyscript import Element

word_element = Element("word")
definition_element = Element("definition")
ai_score_element = Element("ai-score")
player_score_element = Element("player-score")
response_element = Element("response")

player_score:int = 0
ai_score:int = 0
is_real_word:bool = True
word:str = "balderdash"

def read_csv(file_path: str) -> List[str]:
    """A function to load csv

    Args:
        file_path (str): A path to the file

    Returns:
        List[str]: The CSV rows
    """
    # init
    data = []
    # load
    with open(file_path, "r") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            data.append(list(row.values()))
    return data

balder_dash_words = read_csv("./assets/data/balderdash_words.csv")


def word_guess(selection:bool):
    global word_element, definition_element, ai_score_element, player_score_element, response_element,  player_score, ai_score, is_real_word, word, balder_dash_words

    span_title:str = ""
    if selection == is_real_word:
        player_score += 1
        span_title = '<span class="right">CORRECT</span> :'
    else:
        ai_score += 1
        span_title = '<span class="wrong">INCORRECT</span> :'
    span_title += f' {word.capitalize()} is {"not" if is_real_word == False else ""} a real word!'

    word_entry = balder_dash_words[random.randint(0,len(balder_dash_words)-1)]

    word_element.element.innerHTML = word_entry[0].capitalize()
    word = word_entry[0].capitalize()
    definition_element.element.innerHTML = word_entry[1].capitalize()
    is_real_word = (word_entry[2] == "TRUE")
    ai_score_element.element.innerHTML = ai_score
    player_score_element.element.innerHTML = player_score
    response_element.element.innerHTML = span_title

    
