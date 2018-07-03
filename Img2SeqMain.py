import os
import numpy as np
from keras_contrib.utils import save_load_utils
import LoadData
import Models
import Training

# global variables for threading
EMBEDDING_SIZE = 300
SENTENCE_LENGTH = 16
FREQUENCY_OF_WORDS_NEEDED = 8

SENTENCE_START_SYMBOL = '*S*'
SENTENCE_END_SYMBOL = '*E*'
UNKNOWN_SYMBOL = 'ukn'


class Agent:
    def __init__(self, name, train_dataset, val_dataset, w2v_model_number):
        np.random.seed(7)

        self.name = name

        self.loaded_data = LoadData.LoadData(w2v_model_number, SENTENCE_LENGTH, FREQUENCY_OF_WORDS_NEEDED, EMBEDDING_SIZE, train_dataset, val_dataset)

        # get size of state and action, and inputs
        self.state_space = [SENTENCE_LENGTH, EMBEDDING_SIZE]
        self.action_size = self.loaded_data.action_size
        self.image_vector_size = self.loaded_data.image_vector_size

        # create model for actor network
        self.actor = Models.build_actor_model(self.state_space, self.action_size, self.image_vector_size)

    def train_model(self, data_partition=10000, epochs=35, batch_size=64, validation=False, validation_k=30):
        trainer = Training.Training(self.actor, self.loaded_data, self.name)
        trainer.train(data_partition=data_partition, batch_size=batch_size, epochs=epochs, validation=validation, validation_k=validation_k)
        save_model(self.actor, 'actor_' + self.name)

    def validate_model(self, k = 30):
        validator = Training.Training(self.actor, self.loaded_data, self.name)
        validator.validate(k)


    def load_actor(self, path):
        if (os.path.isfile(path)):
            save_load_utils.load_all_weights(self.actor, path)
            print('Weights from old actor model found and loaded')

def save_model(model, name):
    save_load_utils.save_all_weights(model, './save_model/' + name + '.model')

if __name__ == "__main__":
    name = "Img2SeqTest01"
    train_dataset = True
    train_epochs = 35

    val_dataset = True
    val_k = 30
    w2vModel = 0

    agent = Agent(name, train_dataset, val_dataset, w2vModel)

    # agent.load_actor(r'./save_model/actor_Seq2seq.model')

    agent.train_model(epochs=train_epochs, validation=True, validation_k=val_k)
    # agent.validate_model()