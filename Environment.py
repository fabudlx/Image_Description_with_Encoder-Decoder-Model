import random
from collections import Counter
import nltk
import os
import pickle
import numpy as np
from keras import preprocessing
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

import EnvironmentState
import Models


class SentenceDiscriminator:

    def __init__(self, name, w2v_class, sentence_length, frequency_of_words_needed, embedding_size, dataset, w2v_model_name, discriminator_lr):

        self.sentence_length = sentence_length
        self.frequency_of_words_needed = frequency_of_words_needed

        self.name = name
        self.w2v_class = w2v_class
        self.id_to_vector_dict, self.id_to_caption_dict = self.get_image_dictionaries(dataset=dataset)
        self.image_vector_size = len(list(self.id_to_vector_dict.values())[0])
        self.list_of_ids = list(self.id_to_vector_dict.keys())

        self.integer_to_word_dict = self.get_actions()
        self.id_to_embedded_captions = self.get_id_to_embedded_captions(path='./save_model/id_to_embedded_captions_'+w2v_model_name+'_'+dataset+'.dict')
        print('sätze:'+str(len(self.id_to_embedded_captions)))

        self.action_size = len(self.integer_to_word_dict)
        self.state_space = [sentence_length, embedding_size]
        self.discriminator = Models.get_discriminator(self.state_space, self.image_vector_size)

        self.states = []
        self.discriminator_predictions = {'positive': [], 'negative': []}


    def get_image_dictionaries(self, path=r'./save_model', dataset = 'train'):
        with open(os.path.join(path, 'id_to_vector_'+dataset+'.dict'), 'rb') as file:
            vector_dic = pickle.loads(file.read())
        with open(os.path.join(path, 'id_to_caption_'+dataset+'.dict'), 'rb') as file:
            caption_dic = pickle.loads(file.read())
        return vector_dic, caption_dic


    def get_new_state(self):
        environment_state = EnvironmentState.EnvironmentState(self)
        self.states.append(environment_state)
        return environment_state


    def stop_state(self, state):
        self.states.remove(state)


    def train(self, negative_samples, epochs = 1):
        # fake/negative: 0
        # real/positive: 1
        positive_samples =[]
        for i in range(len(negative_samples)):
            image_id = random.choice(self.list_of_ids)
            positive_samples.append([random.choice(self.id_to_embedded_captions[image_id]), self.id_to_vector_dict[image_id]])

        self.train_discriminator(epochs, negative_samples, positive_samples)

    def train_with_other_captions(self, no_of_images, runs = 1, training_iterations = 1):

        for _ in range(runs):
            positive_samples = []
            negative_samples = []
            for i in range(no_of_images):
                image_id = random.choice(self.list_of_ids)
                image_vector = self.id_to_vector_dict[image_id]

                for caption in self.id_to_embedded_captions[image_id]:
                    positive_samples.append([caption, image_vector])

                for _ in range(len(self.id_to_embedded_captions[image_id])):
                    different_image_id = random.choice(self.list_of_ids)
                    while different_image_id == image_id:
                        different_image_id = random.choice(self.list_of_ids)
                    negative_samples.append([random.choice(self.id_to_embedded_captions[different_image_id]), image_vector])

            self.train_discriminator(training_iterations, negative_samples, positive_samples)

    def train_discriminator(self, training_iterations, negative_samples, positive_samples):
        y_pos = np.ones((len(positive_samples), 1))
        # noise = np.random.random((len(y_pos), 1)) * 0.1
        # y_pos -= noise

        y_neg = np.zeros((len(negative_samples), 1))
        # noise = np.random.random((len(y_neg), 1)) * 0.1
        # y_neg += noise

        pos_x1 = np.array([sample[0] for sample in positive_samples])
        pos_x2 = np.array([sample[1] for sample in positive_samples])
        neg_x1 = np.array([sample[0] for sample in negative_samples])
        neg_x2 = np.array([sample[1] for sample in negative_samples])

        self.discriminator.fit([pos_x1, pos_x2], y_pos, epochs= training_iterations, verbose=2)
        self.discriminator.fit([neg_x1, neg_x2], y_neg, epochs=training_iterations, verbose=2)



        # positive_predict = np.mean(self.discriminator.predict([pos_x1, pos_x2]), axis=0)
        # negative_predict = np.mean(self.discriminator.predict([neg_x1, neg_x2]), axis=0)

        # print("(BEFORE) positive prediction belive: " + str(positive_predict[0]) + " negative prediction belive: " + str(negative_predict[0]))

        # self.discriminator_predictions['positive'].append(positive_predict[0])
        # self.discriminator_predictions['negative'].append(negative_predict[0])
        #
        # with open("./save_graph/" + self.name + "_discriminator_scores.bin", "wb") as score_file:
        #     pickle.dump(self.discriminator_predictions, score_file)

        # positive_predict = np.mean(self.discriminator.predict([pos_x1, pos_x2]), axis=0)
        # negative_predict = np.mean(self.discriminator.predict([neg_x1, neg_x2]), axis=0)
        # print("(AFTER) positive prediction belive: " + str(positive_predict[0]) + " negative prediction belive: " + str(negative_predict[0]))


    def integer_to_embedding(self, integer_list_of_sentences):
        return [self.w2v_class.get_embeddings([self.integer_to_word_dict[integer] for integer in integer_sentence]) for integer_sentence in integer_list_of_sentences]


    def get_actions(self):

        # puncs = set(string.punctuation)
        # puncs.add('``')
        # puncs.add("''")
        # tokenized_sentences = [[word for word in sentence if word not in puncs] for sentence in tokenized_sentences]

        all_captions = [nltk.word_tokenize(sentence) for sentences in list(self.id_to_caption_dict.values()) for sentence in sentences]

        all_captions = [[token.lower() for token in sentence] for sentence in all_captions]
        all_tokens = [token for sentence in all_captions for token in sentence]
        counter = Counter(all_tokens)
        # removing all words that are not in the reduced counter -> having a lower frequency than FREQUENCY_OF_WORDS_NEEDED
        reduced_counter = {word: value for word, value in counter.items() if value > self.frequency_of_words_needed}

        # integer encode
        list_of_words = list(reduced_counter.keys())
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(list_of_words)
        integer_to_word_dict = dict(zip(integer_encoded, list_of_words))
        integer_to_word_dict[len(integer_to_word_dict)] = "*S*"
        integer_to_word_dict[len(integer_to_word_dict)] = "*E*"
        integer_to_word_dict[len(integer_to_word_dict)] = "*ukn*"

        print("Corpus contains " + str(len(all_captions)) + " sentences.\n")
        print("\nReading corpus: accepting every word with at least a frequency of " + str(self.frequency_of_words_needed) + " yields " + str(len(reduced_counter)) + " words/actions.\n")

        return integer_to_word_dict

    def get_id_to_embedded_captions(self, path):

        if os.path.isfile(path):
            with open(path, 'rb') as file:
                id_to_embedded_captions = pickle.loads(file.read())
        else:
            id_to_embedded_captions = defaultdict(list)
            for id, captions in self.id_to_caption_dict.items():
                embedded_captions_list = []
                for caption in captions:
                    tokenized_caption = nltk.word_tokenize(caption)
                    embedded_captions_list.append(self.w2v_class.get_embeddings(tokenized_caption, start_token=True))

                id_to_embedded_captions[id] = preprocessing.sequence.pad_sequences(embedded_captions_list, maxlen=self.sentence_length, dtype='float32', padding='post', truncating='post', value=0.)

            with open(path, 'wb') as file:
                pickle.dump(id_to_embedded_captions, file, protocol=4)

        return id_to_embedded_captions


