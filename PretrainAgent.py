import SingleAgent
from ImageDescriptionGradientPolicy import SENTENCE_LENGTH, EMBEDDING_SIZE
import nltk
from keras.utils import to_categorical
import numpy as np
from keras import preprocessing
import pickle
import random
from keras_contrib.utils import save_load_utils


class PretrainAgent(SingleAgent.Agent):


    def __init__(self, actor, env, id_to_vector_dict, discount_factor, action_size, state_size, random_input_size, scores, episode):
        super().__init__(actor, env, id_to_vector_dict, discount_factor, action_size, state_size, random_input_size, scores, episode)

        self.id_to_embedded_captions = env.id_to_embedded_captions
        self.id_to_caption_dict = env.id_to_caption_dict
        self.integer_to_word_dict = env.integer_to_word_dict
        self.correct_sentence_as_list = []
        self.word_to_integer_dict = {value: key for key, value in self.integer_to_word_dict.items()}
        self.agent_type = 'pretrain'

    def pretrain(self, data_partition = 10000, batch_size = 64, epochs = 20):

        image_vectors, decoder_input, decoder_target = [], [], []

        for no, image_id in enumerate(self.list_of_ids):

            list_of_captions_as_word_lists = [nltk.word_tokenize(sentence.lower()) for sentence in self.id_to_caption_dict[image_id]]
            list_of_embedded_captions = self.id_to_embedded_captions[image_id]


            for caption_as_word_list, embedded_caption in zip(list_of_captions_as_word_lists, list_of_embedded_captions):

                sentence_as_integer_list = []
                for word in caption_as_word_list:
                    if word in self.word_to_integer_dict:
                        sentence_as_integer_list.append(self.word_to_integer_dict[word])
                    else:
                        sentence_as_integer_list.append(len(self.word_to_integer_dict) - 1) #UNKNOWN
                sentence_as_integer_list.append(len(self.word_to_integer_dict) - 2) #END

                sentence_as_hot_one_encoding = to_categorical(sentence_as_integer_list, num_classes=len(self.integer_to_word_dict))

                image_vectors.append(self.id_to_vector_dict[image_id])
                decoder_input.append(embedded_caption)
                decoder_target.append(sentence_as_hot_one_encoding)

                # print(self.actor.predict([np.array(image_vectors), np.array(decoder_input)]))

            if no is not 0 and no % 1000 == 0:
                print(str(len(image_vectors))+' samples have been created')

            if no is not 0 and no % data_partition == 0:
                decoder_target = preprocessing.sequence.pad_sequences(decoder_target, maxlen=SENTENCE_LENGTH, dtype='int16', padding='post', truncating='post', value=0)
                print(str(len(image_vectors))+' samples have been created, Actor will be trained for '+str(epochs)+' epochs, with batch_size of '+str(batch_size))
                # verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
                history_callback = self.actor.fit([np.array(image_vectors), np.array(decoder_input)], np.array(decoder_target), verbose=0, batch_size=batch_size, epochs = epochs)
                image_vectors, decoder_input, decoder_target = [], [], []

                with open('./save_graph/seq2seq_scores_'+str(no)+'.bin', 'wb') as result_dict:
                    pickle.dump(history_callback.history, result_dict)
                save_load_utils.save_all_weights(self.actor, './save_model/actor_Seq2seq.model')

                #Test
                k = 10
                random_image_ids = random.sample(self.list_of_ids, k = k)

                test_image_vectors = np.array([self.id_to_vector_dict[image_id] for image_id in random_image_ids])

                sentence_vectors = np.zeros((k, SENTENCE_LENGTH, EMBEDDING_SIZE))
                sentence_vectors[:, 0] = 1

                for i in range(SENTENCE_LENGTH-1):
                    global predictions
                    predictions = self.actor.predict([test_image_vectors, sentence_vectors])
                    predicted_words_as_hot_one = [prediction[i] for prediction in predictions]
                    predicted_words_as_integers = np.argmax(predicted_words_as_hot_one, axis=1)
                    predicted_words = [self.integer_to_word_dict[integer] for integer in predicted_words_as_integers]
                    predicted_embedded_words = self.env.w2v_class.get_embeddings(predicted_words)

                    sentence_vectors[:, i+1] = predicted_embedded_words

                predicted_sentences_as_integers = np.argmax(predictions, axis=2)
                for sentences, image_id in zip(predicted_sentences_as_integers, random_image_ids):
                    print('Image '+str(image_id)+': '+' '.join([self.integer_to_word_dict[integer] for integer in sentences]))
