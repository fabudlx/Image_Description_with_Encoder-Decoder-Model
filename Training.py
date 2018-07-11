import Img2SeqMain
from keras.utils import to_categorical
import numpy as np
import keras.preprocessing.sequence
import pickle, nltk, random
import nltk.translate.bleu_score
import logging, json, os

logger = logging.getLogger("_logger_")

class Training():
    def __init__(self, actor, loaded_data, name, result_folder):



        self.actor = actor
        self.loaded_data = loaded_data
        self.name = name

        self.result_folder = result_folder
        self.history_folder = os.path.join(self.result_folder, 'training_history')
        self.result_sentences_folder = os.path.join(self.result_folder, 'training_results')

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
                logger.info(str(len(image_vectors))+' samples have been created')

            if no is not 0 and no % data_partition == 0:
                decoder_target = keras.preprocessing.sequence.pad_sequences(decoder_target, maxlen=Img2SeqMain.SENTENCE_LENGTH, dtype='int16', padding='post', truncating='post', value=0)
                logger.info(str(len(image_vectors))+' samples have been created, Actor will be trained for '+str(epochs)+' epochs, with batch_size of '+str(batch_size))
                # verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
                history_callback = self.actor.fit([np.array(image_vectors), np.array(decoder_input)], np.array(decoder_target), verbose=0, batch_size=batch_size, epochs = epochs)
                image_vectors, decoder_input, decoder_target = [], [], []

                if not os.path.isdir(self.history_folder):
                    os.makedirs(self.history_folder)

                history_file = os.path.join(self.history_folder, 'history_'+str(no)+'.history')
                with open(history_file, 'wb') as result_dict:
                    pickle.dump(history_callback.history, result_dict, protocol=2)
                    logger.info('saving history under '+ history_file)
                Img2SeqMain.save_model(self.actor, self.name, self.result_folder)

                if validation:
                    self.validate(validation_k, no)


    def validate(self, k, no = -1):

        if k >= len(self.loaded_data.list_of_val_image_ids):
            image_ids_for_validation = self.loaded_data.list_of_val_image_ids
        else:
            image_ids_for_validation = random.sample(self.loaded_data.list_of_val_image_ids, k=k)
        test_image_vectors = np.array([self.loaded_data.val_id_to_image_vector_dict[image_id] for image_id in image_ids_for_validation])
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
                sentences_cut_after_eos.append(sentence[:sentence.index(Img2SeqMain.SENTENCE_END_SYMBOL)])
            else:
                sentences_cut_after_eos.append(sentence)

        results = []
        for hypothesis, image_id in zip(sentences_cut_after_eos, image_ids_for_validation):

            # "image_id" : int, "caption" : str,
            results.append({'image_id': image_id, 'caption': ' '.join(hypothesis)})

        if not os.path.isdir(self.result_sentences_folder):
            os.makedirs(self.result_sentences_folder)

        if no > 0:
            name = self.result_sentences_folder+'/results_after'+str(no)+'_images.json'
        else:
            name = self.result_sentences_folder+'/final_results.json'

        with open(name, 'w') as fp:
            json.dump(results, fp, protocol=2)
            logger.info('saving results for '+str(k) +' images under '+name)


        #
        #     bleu1 = nltk.translate.bleu_score.sentence_bleu(references, hypothesis, weights=(1, 0, 0, 0))
        #     bleu2 = nltk.translate.bleu_score.sentence_bleu(references, hypothesis, weights=(0.5, 0.5, 0, 0))
        #     bleu3 = nltk.translate.bleu_score.sentence_bleu(references, hypothesis, weights=(0.33, 0.33, 0.33, 0))
        #     bleu4 = nltk.translate.bleu_score.sentence_bleu(references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25))
        #     logger.info('Image ' + str(image_id) + ': ' + ' '.join(hypothesis)+' BLEU-1 Score: '+str(bleu1)+' BLEU-2 Score: '+str(bleu2)+' BLEU-3 Score: '+str(bleu3)+' BLEU-4 Score: '+str(bleu4))
