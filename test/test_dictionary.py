import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

from balderdash_lstm.dictionary.main import get_random_word_of_the_day, get_word_definition


class DictionaryTestCase(unittest.TestCase):

    def test_get_word_definition(self):
        # Test case 1: Valid word with definition
        word1 = 'hello'
        success1, definition1 = get_word_definition(word1)
        assert success1 == True
        assert definition1 != ""

        # Test case 2: Invalid word
        word2 = 'asdfghjkl'
        success3, definition3 = get_word_definition(word2)
        assert success3 == False
        assert definition3 == f"Error 404: Unable to fetch definition for '{word2}'."
    
    def test_get_random_word_of_the_day_word_found(self):
        result = get_random_word_of_the_day()
        assert isinstance(result, tuple)
        assert result[0] is True
        assert isinstance(result[1], str)


if __name__ == "__main__":
    unittest.main()
