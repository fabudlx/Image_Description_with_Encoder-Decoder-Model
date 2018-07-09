import os
import numpy as np
from keras_contrib.utils import save_load_utils
import LoadData
import Models
import Training
import logging
import datetime
import time, sys
import pickle
# global variables for threading
EMBEDDING_SIZE = 300
SENTENCE_LENGTH = 16
FREQUENCY_OF_WORDS_NEEDED = 8

SENTENCE_START_SYMBOL = '*S*'
SENTENCE_END_SYMBOL = '*E*'
UNKNOWN_SYMBOL = 'ukn'


logger = logging.getLogger("_logger_")

def log_exceptions(exctype, value, tb):
    logger.error("Uncaught exception", exc_info=(exctype, value, tb))


class Img2Seq:
    def __init__(self, name, train_dataset, val_dataset, w2v_model_number):

        np.random.seed(7)

        self.name = name

        timestamp = time.time()
        date_time = str(datetime.datetime.fromtimestamp(timestamp).strftime('%d%m%Y-%H%M'))

        self.result_folder = r'./ExperimentResults/'+date_time+'_'+self.name

        os.makedirs(self.result_folder)



        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            filename=self.result_folder+'/info.log',
                            filemode='w')
        sys.excepthook = log_exceptions


        self.loaded_data = LoadData.LoadData(w2v_model_number, SENTENCE_LENGTH, FREQUENCY_OF_WORDS_NEEDED, EMBEDDING_SIZE, train_dataset, val_dataset, self.result_folder)

        # get size of state and action, and inputs
        self.state_space = [SENTENCE_LENGTH, EMBEDDING_SIZE]
        self.action_size = self.loaded_data.action_size
        self.image_vector_size = self.loaded_data.image_vector_size

        # create model for actor network
        self.model = Models.build_actor_model(self.state_space, self.action_size, self.image_vector_size)

        with open(self.result_folder + '/model_architecture.txt', 'w') as fh:
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))
        with open(self.result_folder + '/model_config.bin', 'wb') as fh:
            pickle.dump(self.model.get_config(), fh)

    def train_model(self, data_partition=20000, epochs=35, batch_size=64, validation=False, validation_k=30):
        trainer = Training.Training(self.model, self.loaded_data, self.name, self.result_folder)
        trainer.train(data_partition=data_partition, batch_size=batch_size, epochs=epochs, validation=validation, validation_k=validation_k)
        save_model(self.model, self.name, self.result_folder)

    def validate_model(self, k = 30):
        validator = Training.Training(self.model, self.loaded_data, self.name, self.result_folder)
        validator.validate(k)

    def load_actor(self, timestamp, name):

        path = os.path.join(r'./ExperimentResults', timestamp+'_'+name, name+'.model')
        if (os.path.isfile(path)):
            save_load_utils.load_all_weights(self.model, path)
            logger.info('Weights from '+path+' found and loaded')
        else:
            logger.info('Weights from '+path+' could NOT be found and loaded')

def save_model(model, name, result_folder):
    path = os.path.join(result_folder, name + '.model')
    save_load_utils.save_all_weights(model, path)
    logger.info('saving model under '+path)



def main():

    if len(sys.argv) >= 2:
        train_epochs = int(sys.argv[1])
        if train_epochs > 0:
            train_dataset = True
        else:
            train_dataset = False
    else:
        train_epochs=10
        train_dataset = True

    if len(sys.argv) >= 3:
        val_epochs = int(sys.argv[2])
        if val_epochs > 0:
            val_dataset = True
        else:
            val_dataset = False
    else:
        val_epochs = 50
        val_dataset = True

    if len(sys.argv) >= 4:
        w2vModel = int(sys.argv[3])
    else:
        w2vModel = 0

    if len(sys.argv) >= 5:
        data_partition = int(sys.argv[4])
    else:
        data_partition = 20000

    name = ''

    if train_dataset:
        name = name+str(train_epochs)+'_tra_epos'
    if val_dataset:
        name = name+'_'+str(val_epochs)+'_val_imgs'
    name = name+'_'+str(w2vModel)+'_w2v'
    name = name + '_' + str(data_partition) + 'dat_part'

    agent = Img2Seq(name, train_dataset, val_dataset, w2vModel)

    # agent.load_actor('04072018-1630', name)
    if train_dataset:
        agent.train_model(epochs=train_epochs, validation=val_dataset, validation_k=val_epochs, data_partition=data_partition)
    if val_dataset:
        agent.validate_model(val_epochs)


if __name__ == "__main__":
    main()