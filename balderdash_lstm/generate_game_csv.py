from lstm_model.train import load_models
from lstm_model.utils import get_tokens
from lstm_model.lstm import predict
from dictionary.main import get_random_word_of_the_day, get_word_definition

if __name__ == "__main__":
    """
    Due to CORS, I am not able to have the website pull this data live. While sadden as I learned this at the very end of the project, 
    this was the best solution I could find! With at least 500 words, that should be enough variation for the regular
    user to get to have fun! In addition, if anyone wants to pull this repo and use different words to train their model on,
    they should be able to re-run this code and create a new csv that is usuable.
    """

    csv_name = "balderdash_words.csv"

    csv_data = 'word, definition, is_real_word\n'

    lstm, optimizer, embeddings = load_models('./lstm_model/models/lstm_model_25.npz')  
    tokens, tokens_to_id, id_to_tokens =  get_tokens()

    for _ in range(500):
        word = ''
        while not word:
            word = predict(lstm=lstm, embeddings=embeddings, token_id_mapping=id_to_tokens,tokens_size=len(tokens),end_token=".",itterations=1)[0]
            word.replace(".",'')
        
        is_real_word, definition = get_word_definition(word)
        if not is_real_word:
            _, random_world = get_random_word_of_the_day()
            _, definition = get_word_definition(random_world)
            print(definition)
            definition = definition[19 + len(random_world):].split('.')[0]
        else:
            print(definition)
            definition = definition[18 + len(word):].split('.')[0]
        
        csv_data += f'{word[:-1]},"{definition}",{is_real_word}\n'
            
    csv_file = open(csv_name,"w")
    csv_file.write(csv_data)