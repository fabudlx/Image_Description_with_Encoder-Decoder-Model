from collections import Counter
import nltk, os, pickle
import keras.preprocessing.sequence
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from WordEmbedding import EmbeddingModel

from Img2SeqMain import SENTENCE_END_SYMBOL, SENTENCE_START_SYMBOL, UNKNOWN_SYMBOL


def load_w2v(model_name, embedding_size):
    if model_name is 0:
        w2v_class = EmbeddingModel(embedding_size, path=r'../Automated-Image-Description/files/w2vModel/wiki.en.vec', lower=True, binary=False)
        return w2v_class,'fasttext'
    elif model_name is 1:
        w2v_class = EmbeddingModel(embedding_size, path=r'../Automated-Image-Description/files/w2vModel/google_model_en.bin', lower=True, binary=True)
        return w2v_class, 'google'
    else:
        w2v_class = EmbeddingModel(embedding_size, r'../Automated-Image-Description/files/w2vModel/en.wiki.bpe.op200000.d300.w2v.bin', lower=True, binary=True)
        return w2v_class, 'bpe'

class LoadData:

    def __init__(self, w2v_model_number, sentence_length, frequency_of_words_needed, embedding_size, train_dataset, val_dataset):

        self.sentence_length = sentence_length
        self.frequency_of_words_needed = frequency_of_words_needed
        self.embedding_size = embedding_size

        #load word to vector model
        self.w2v_class, w2v_model_name = load_w2v(w2v_model_number, self.embedding_size)



        #Traing Data
        #load preprocessed Image vectors (basically last layer of CNN) and preprocessed Captions
        self.train_id_to_image_vector_dict, self.train_id_to_caption_dict = self.get_image_and_caption_dicts(dataset=train_dataset)
        #all id's of all images
        self.list_of_train_image_ids = list(self.train_id_to_image_vector_dict.keys())
        self.train_id_to_embedded_captions = self.get_id_to_embedded_captions(path='./save_model/id_to_embedded_captions_' + w2v_model_name + '_' + train_dataset + '.dict')
        print('No of training sentences:' + str(len(self.train_id_to_embedded_captions)))

        #Same for Validation Data
        self.val_id_to_image_vector_dict, self.val_id_to_caption_dict = self.get_image_and_caption_dicts(dataset=val_dataset)
        self.list_of_val_image_ids = list(self.val_id_to_image_vector_dict.keys())
        self.val_id_to_embedded_captions = self.get_id_to_embedded_captions(path='./save_model/id_to_embedded_captions_' + w2v_model_name + '_' + val_dataset + '.dict')
        print('No of validation sentences:' + str(len(self.val_id_to_embedded_captions)))

        #dictionary mapping all words to integers
        self.integer_to_word_dict = self.get_actions()
        self.action_size = len(self.integer_to_word_dict)

        #Size of image vectors
        self.image_vector_size = len(list(self.train_id_to_image_vector_dict.values())[0])

    def get_image_and_caption_dicts(self, path=r'./save_model', dataset ='train'):
        with open(os.path.join(path, 'id_to_vector_'+dataset+'.dict'), 'rb') as file:
            id_to_vector_dict = pickle.loads(file.read())
        with open(os.path.join(path, 'id_to_caption_'+dataset+'.dict'), 'rb') as file:
            id_to_caption_dict = pickle.loads(file.read())
        return id_to_vector_dict, id_to_caption_dict

    def get_actions(self):
        all_captions = [nltk.word_tokenize(sentence) for sentences in list(self.train_id_to_caption_dict.values()) for sentence in sentences]
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
        integer_to_word_dict[len(integer_to_word_dict)] = SENTENCE_START_SYMBOL #len(integer_to_word_dict) -3
        integer_to_word_dict[len(integer_to_word_dict)] = SENTENCE_END_SYMBOL #len(integer_to_word_dict) -2
        integer_to_word_dict[len(integer_to_word_dict)] = UNKNOWN_SYMBOL #len(integer_to_word_dict) -1

        print("Corpus contains " + str(len(all_captions)) + " sentences.\n")
        print("\nReading corpus: accepting every word with at least a frequency of " + str(self.frequency_of_words_needed) + " yields " + str(len(reduced_counter)) + " words/actions.\n")

        return integer_to_word_dict

    def get_id_to_embedded_captions(self, path):
        if os.path.isfile(path):
            print('Found ID_to_Embedding file '+path+'\nLoading file now!')
            with open(path, 'rb') as file:
                id_to_embedded_captions = pickle.loads(file.read())
        else:
            print('NO file found under' + path + '\nCreating file now')
            id_to_embedded_captions = defaultdict(list)
            for image_id, captions in self.train_id_to_caption_dict.items():
                embedded_captions_list = []
                for caption in captions:
                    tokenized_caption = nltk.word_tokenize(caption)
                    embedded_captions_list.append(self.w2v_class.get_embeddings(tokenized_caption, start_token=True, split_words=True))

                id_to_embedded_captions[image_id] = keras.preprocessing.sequence.pad_sequences(embedded_captions_list, maxlen=self.sentence_length, dtype='float32', padding='post', truncating='post', value=0.)
            print('ID_to_Embedding dictionary has been created. Saving now under ' + path)
            with open(path, 'wb') as file:
                pickle.dump(id_to_embedded_captions, file, protocol=4)
                print('Successfully saved!')
        return id_to_embedded_captions

    def integer_to_embedding(self, integer_list_of_sentences):
        return [self.w2v_class.get_embeddings([self.integer_to_word_dict[integer] for integer in integer_sentence]) for integer_sentence in integer_list_of_sentences]