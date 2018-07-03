import Img2SeqMain
from keras.utils import to_categorical
import numpy as np
import keras.preprocessing.sequence
import pickle, nltk, random
from keras_contrib.utils import save_load_utils
import nltk.translate.bleu_score

class Training():
    def __init__(self, actor, loaded_data, name):

        self.actor = actor
        self.loaded_data = loaded_data
        self.name = name

        #dictionary mapping either words to interger values or vice-versa
        self.integer_to_word_dict = loaded_data.integer_to_word_dict
        self.word_to_integer_dict = {value: key for key, value in self.integer_to_word_dict.items()}

    def train(self, data_partition, batch_size, epochs, validation, validation_k):

        image_vectors, decoder_input, decoder_target = [], [], []

        for no, image_id in enumerate(self.loaded_data.list_of_train_image_ids):

            list_of_captions_as_word_lists = [nltk.word_tokenize(sentence.lower()) for sentence in self.loaded_data.train_id_to_caption_dict[image_id]]
            list_of_embedded_captions = self.loaded_data.train_id_to_embedded_captions[image_id]


            for caption_as_word_list, embedded_caption in zip(list_of_captions_as_word_lists, list_of_embedded_captions):

                sentence_as_integer_list = []
                for word in caption_as_word_list:
                    if word in self.word_to_integer_dict:
                        sentence_as_integer_list.append(self.word_to_integer_dict[word])
                    else:
                        sentence_as_integer_list.append(len(self.word_to_integer_dict) - 1) #UNKNOWN
                sentence_as_integer_list.append(len(self.word_to_integer_dict) - 2) #END

                sentence_as_hot_one_encoding = to_categorical(sentence_as_integer_list, num_classes=len(self.integer_to_word_dict))

                image_vectors.append(self.loaded_data.train_id_to_image_vector_dict[image_id])
                decoder_input.append(embedded_caption)
                decoder_target.append(sentence_as_hot_one_encoding)

                # print(self.actor.predict([np.array(image_vectors), np.array(decoder_input)]))

            if no is not 0 and no % 1000 == 0:
                print(str(len(image_vectors))+' samples have been created')

            if no is not 0 and no % data_partition == 0:
                decoder_target = keras.preprocessing.sequence.pad_sequences(decoder_target, maxlen=Img2SeqMain.SENTENCE_LENGTH, dtype='int16', padding='post', truncating='post', value=0)
                print(str(len(image_vectors))+' samples have been created, Actor will be trained for '+str(epochs)+' epochs, with batch_size of '+str(batch_size))
                # verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
                history_callback = self.actor.fit([np.array(image_vectors), np.array(decoder_input)], np.array(decoder_target), verbose=0, batch_size=batch_size, epochs = epochs)
                image_vectors, decoder_input, decoder_target = [], [], []

                with open('./save_graph/seq2seq_scores_'+str(no)+'.bin', 'wb') as result_dict:
                    pickle.dump(history_callback.history, result_dict)
                save_load_utils.save_all_weights(self.actor, './save_model/actor_'+self.name+'.model')

                if validation:
                    self.validate(validation_k)


    def validate(self, k):
        # Test
        random_image_ids = random.sample(self.loaded_data.list_of_val_image_ids, k=k)
        test_image_vectors = np.array([self.loaded_data.val_id_to_image_vector_dict[image_id] for image_id in random_image_ids])
        sentence_vectors = np.zeros((k, Img2SeqMain.SENTENCE_LENGTH, Img2SeqMain.EMBEDDING_SIZE))
        sentence_vectors[:, 0] = 1
        for i in range(Img2SeqMain.SENTENCE_LENGTH - 1):
            global prediction
            prediction = self.actor.predict([test_image_vectors, sentence_vectors])
            predicted_words_as_hot_one = [prediction[i] for prediction in prediction]
            predicted_words_as_integers = np.argmax(predicted_words_as_hot_one, axis=1)
            predicted_words = [self.integer_to_word_dict[integer] for integer in predicted_words_as_integers]

            predicted_embedded_word = self.loaded_data.w2v_class.get_embeddings(predicted_words)
            sentence_vectors[:, i + 1] = predicted_embedded_word

        predicted_sentences_as_integers = np.argmax(prediction, axis=2)
        predicted_sentences_as_words = [[self.integer_to_word_dict[integer] for integer in sentences] for sentences in predicted_sentences_as_integers]

        sentences_cut_after_eos = []
        for sentence in predicted_sentences_as_words:
            if Img2SeqMain.SENTENCE_END_SYMBOL in sentence:
                sentences_cut_after_eos.append(sentence[:sentence.index(Img2SeqMain.SENTENCE_END_SYMBOL) + 1])
            else:
                sentences_cut_after_eos.append(sentence)

        for sentence, image_id in zip(sentences_cut_after_eos, random_image_ids):

            bleu_score = nltk.translate.bleu_score.sentence_bleu(self.loaded_data.val_id_to_caption_dict[image_id], sentence)

            print('Image ' + str(image_id) + ': ' + ' '.join(sentence)+' BLEU Score: '+str(bleu_score))
