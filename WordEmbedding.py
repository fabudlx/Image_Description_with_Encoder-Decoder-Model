import re

import numpy as np
from gensim.models.keyedvectors import KeyedVectors

class EmbeddingModel():

    def __init__(self, embedding_size, path = r'./files/w2vModel/wiki.en.vec',lower=False, binary=False):
        print('loading '+path+' ... this could take a while')
        self.w2v_model = KeyedVectors.load_word2vec_format(path, binary=binary)
        self.lower = lower
        self.embedding_size = embedding_size
        print('Model loaded, yeay :)')

    def get_embeddings(self, list_of_words, start_token=False, split_words=False):
        if self.lower:
            list_of_words = [word.lower() for word in list_of_words]

        return_list = []
        not_known_words =[]

        if start_token:
            #embedded start token contains only 1s
            return_list.append(np.ones(self.embedding_size))

        while list_of_words:

            word = list_of_words.pop(0)

            #word is known
            if word in self.w2v_model.vocab:
                return_list.append(self.w2v_model[word])
                continue

            #word is not directly known, let's see if we can find a close match
            not_known_words.append(word.lower())

            #make lower case
            if word.lower() in self.w2v_model.vocab:
                return_list.append(self.w2v_model[word.lower()])
                continue

            #look for numbers
            if bool(re.search(r'\d', word)):
                if '0' in self.w2v_model:
                    return_list.append(self.w2v_model['0'])
                    continue
                if 'one' in self.w2v_model:
                    return_list.append(self.w2v_model['one'])
                    continue
                else:
                    return_list.append(np.ones(self.embedding_size) - 0.8)
                    continue

            #split at '-'
            if split_words and '-' in word:
                words = re.split('-', word)
                list_of_words = words + list_of_words
                continue

            #remove punctiation
            word_without_punctuation = re.sub(r'[^\w\s]', '', word)
            if word_without_punctuation in self.w2v_model.vocab:
                return_list.append(self.w2v_model[word_without_punctuation])
                continue

            #remove punctiation and make lower case
            if word_without_punctuation.lower() in self.w2v_model.vocab:
                return_list.append(self.w2v_model[word_without_punctuation.lower()])
                continue

            #if word is still unkown it is embedded in -1s (unknown)
            return_list.append(np.ones(self.embedding_size) - 2)

        return return_list
