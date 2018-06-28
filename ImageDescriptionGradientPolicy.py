import os
import pickle

import numpy as np
from keras_contrib.utils import save_load_utils

import Environment
import Models
from WordEmbedding import EmbeddingModel

import PretrainAgent, SingleAgent

# global variables for threading
EMBEDDING_SIZE = 300
SENTENCE_LENGTH = 16
FREQUENCY_OF_WORDS_NEEDED = 8


class A3CAgent:
    def __init__(self, name, dataset, w2vModel):
        np.random.seed(7)

        self.episode = 0
        self.scores = []
        self.name = name

        # these are hyper parameters for the A3C
        self.actor_lr = 0.0001
        self.discriminator_lr = 0.0001
        self.discount_factor = .99

        if w2vModel is 0:
            w2v_class = EmbeddingModel(EMBEDDING_SIZE, path=r'../Automated-Image-Description/files/w2vModel/wiki.en.vec', lower=True, binary=False)
            w2v_model_name = 'fasttext'
        elif w2vModel is 1:
            w2v_class = EmbeddingModel(EMBEDDING_SIZE, path=r'../Automated-Image-Description/files/w2vModel/google_model_en.bin', lower=True, binary=True)
            w2v_model_name = 'google'
        else:
            w2v_class = EmbeddingModel(EMBEDDING_SIZE, r'../Automated-Image-Description/files/w2vModel/en.wiki.bpe.op200000.d300.w2v.bin', lower=True, binary=True)
            w2v_model_name = 'bpe'

        self.env = Environment.SentenceDiscriminator(name=self.name, w2v_class=w2v_class, sentence_length=SENTENCE_LENGTH, frequency_of_words_needed=FREQUENCY_OF_WORDS_NEEDED, embedding_size=EMBEDDING_SIZE, dataset=dataset, w2v_model_name = w2v_model_name, discriminator_lr=self.discriminator_lr)
        self.env_name = self.env.name

        # get size of state and action
        self.state_space = self.env.state_space
        self.action_size = self.env.action_size
        self.image_input_size = self.env.image_vector_size


        # create model for actor and critic network
        self.actor = Models.build_actor_model(self.state_space, self.action_size, self.image_input_size)


    def pretrain_actor(self, minibatch = 128, epochs = 20):
        agent = PretrainAgent.PretrainAgent(self.actor, self.env, self.env.id_to_vector_dict, self.discount_factor, self.action_size, self.state_space, self.image_input_size, self.scores, self.episode)
        agent.pretrain()
        save_load_utils.save_all_weights(self.actor, './save_model/actor_only_pretrain_' + name + '.model')

    def pretrain_discriminator(self, epochs = 5, minibatch = 64):
        agent = SingleAgent.Agent(self.actor, self.env, self.env.id_to_vector_dict, self.discount_factor, self.action_size, self.state_space, self.image_input_size, self.scores, self.episode)
        episodes = int(len(self.env.list_of_ids)/minibatch)*epochs
        print(episodes)
        for _ in range(episodes):
            #generated vs. correct captions
            sentence_image_pairs = agent.get_sentence_image_pairs(int(minibatch/2))
            self.env.train(sentence_image_pairs)

            #image-caption matches vs image-caption NO matches
            self.env.train_with_other_captions(minibatch)

        print('Pretraining Discriminator done')
        save_load_utils.save_all_weights(self.env.discriminator, './save_model/discriminator_only_pretrain_' + name + '.model')

    # make agents(local) and start training
    def train(self):
        agent = SingleAgent.Agent(self.actor, self.env, self.env.id_to_vector_dict, self.discount_factor, self.action_size, self.state_space, self.image_input_size, self.scores, self.episode)
        agent.run()


    def load_model(self, path):
        save_load_utils.load_all_weights(self.actor, path + "_actor.model")
        print('ACTOR and CRITIC weights loaded')

    def load_actor(self, path):
        if (os.path.isfile(path)):
            save_load_utils.load_all_weights(self.actor, path)
            print('Weights from old actor model found and loaded')

    def load_discriminator(self, path):
        if (os.path.isfile(path)):
            save_load_utils.load_all_weights(self.env.discriminator, path)
            print('Weights from old discriminator model found and loaded')

    def load_scores(self, path='./save_graph'):
        with open(path+'/'+self.name+'_scores.bin', 'rb') as result_dict:
            self.scores = pickle.load(result_dict)
            print('scores were loaded')



if __name__ == "__main__":
    name = "Gradient_Policy_Pretrain_Test"
    dataset = 'train'
    w2vModel = 0
    global_agent = A3CAgent(name, dataset, w2vModel)
    # global_agent.load_discriminator(r'./save_model/discriminator_only_pretrain_A3C_Test.model')
    # global_agent.load_actor(r'./save_model/actor_only_pretrain_A3C_Test.model')

    # global_agent.load_discriminator(r'./save_model/A3C_Test_discriminator.model')
    # global_agent.load_model('./save_model/A3C_Test')
    # global_agent.load_scores('./save_graph')

    global_agent.pretrain_actor()
    # global_agent.pretrain_discriminator()
    # global_agent.train()