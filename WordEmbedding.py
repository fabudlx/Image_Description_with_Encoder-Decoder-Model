import re

import numpy as np
from gensim.models.keyedvectors import KeyedVectors

class EmbeddingModel():

    def __init__(self, embedding_size, path = r'./files/w2vModel/wiki.en.vec',lower=False, binary = False):
        print('loading w2v model... this could take a while')
        self.w2v_model = KeyedVectors.load_word2vec_format(path, binary=binary)
        self.lower = lower
        self.embedding_size = embedding_size
        print('word2vec model loaded')

    def get_embeddings(self, list_of_words, start_token = False):
        return_list = []
        not_known_words =[]

        if start_token:
            return_list.append(np.ones(self.embedding_size))
        for word in list_of_words:
            if self.lower:
                word = word.lower()
                if word.lower() in self.w2v_model.vocab:
                    return_list.append(self.w2v_model[word.lower()])
                    continue

                # elif word.lower() == '-empty-':
                #     return_list.append(np.ones(EMBEDDING_SIZE) - 0.5)
                #     continue

                if bool(re.search(r'\d', word)):
                    if '0' in self.w2v_model:
                        return_list.append(self.w2v_model['0'])
                    if 'one' in self.w2v_model:
                        return_list.append(self.w2v_model['one'])
                    else:
                        return_list.append(np.ones(self.embedding_size) - 0.8)
                    continue

                # if '-' in word:
                #     words = re.split('-', word)
                #
                #     for w in words:
                #         if w.lower() in self.w2v_model.vocab:
                #             return_list.append(self.w2v_model[w.lower()])
                #             continue
                #         word_without_punctuation = re.sub(r'[^\w\s]', '', w)
                #         if word_without_punctuation.lower() in self.w2v_model.vocab:
                #             return_list.append(self.w2v_model[word_without_punctuation.lower()])
                #             continue
                #         return_list.append(np.ones(self.embedding_size) - 2)
                #         not_known_words.append(w.lower())
                #     continue

                word_without_punctuation = re.sub(r'[^\w\s]', '', word)
                if word_without_punctuation.lower() in self.w2v_model.vocab:
                    return_list.append(self.w2v_model[word_without_punctuation.lower()])
                    continue

                return_list.append(np.ones(self.embedding_size) - 2)
                not_known_words.append(word.lower())


            else:
                if word in self.w2v_model.vocab:
                    return_list.append(self.w2v_model[word])
                    continue
                if word.lower() in self.w2v_model.vocab:
                    return_list.append(self.w2v_model[word.lower()])
                    continue

                if bool(re.search(r'\d', word)):
                    if '0' in self.w2v_model:
                        return_list.append(self.w2v_model['0'])
                    if 'one' in self.w2v_model:
                        return_list.append(self.w2v_model['one'])
                    else:
                        return_list.append(np.ones(self.embedding_size) - 0.8)
                    continue

                # if '-' in word:
                #     words = re.split('-', word)
                #
                #     for w in words:
                #         if w in self.w2v_model.vocab:
                #             return_list.append(self.w2v_model[w])
                #             continue
                #         if w.lower() in self.w2v_model.vocab:
                #             return_list.append(self.w2v_model[w.lower()])
                #             continue
                #         word_without_punctuation = re.sub(r'[^\w\s]', '', w)
                #         if word_without_punctuation in self.w2v_model.vocab:
                #             return_list.append(self.w2v_model[word_without_punctuation])
                #             continue
                #         if word_without_punctuation.lower() in self.w2v_model.vocab:
                #             return_list.append(self.w2v_model[word_without_punctuation.lower()])
                #             continue
                #         return_list.append(np.ones(self.embedding_size) - 2)
                #         not_known_words.append(w.lower())
                #     continue

                word_without_punctuation = re.sub(r'[^\w\s]', '', word)
                if word_without_punctuation in self.w2v_model.vocab:
                    return_list.append(self.w2v_model[word_without_punctuation])
                    continue

                if word_without_punctuation.lower() in self.w2v_model.vocab:
                    return_list.append(self.w2v_model[word_without_punctuation.lower()])
                    continue

                return_list.append(np.ones(self.embedding_size) - 2)
                not_known_words.append(word.lower())

        return return_list

    def get_words(self, embedding_list):
        pass