import os,torch
import unicodedata
import re
from Voc import Voc
corpus_name = 'cornell movie-dialogs corpus'
corpus = os.path.join('data',corpus_name)

datafile = os.path.join(corpus, "formatted_movie_lines.txt")
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD',s)
        if unicodedata.category(c) != 'Mn'
    )
def normalizeString(s):
    # print(s)
    s = unicodeToAscii(s.lower().strip())
    # print(s)
    s = re.sub(r"([.!?])", r" \1", s)
    # print(s)
    s = re.sub(r"[^a-zA-Z\.!\?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s
def evaluateInput(model,voc,evaluator):
    input_sentence = ''
    while(1):
        try:
            input_sentence = input('> ')
            if input_sentence == 'q' or input_sentence == 'quit': break
            input_sentence = normalizeString(input_sentence)
            output_words = evaluator.evaluate(input_sentence)
            output_words = [x for x in output_words if not(x == 'EOS' or x == 'PAD')]
            print('Bot:',' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


