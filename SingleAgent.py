import threading
import random
import numpy as np
import pickle
from keras_contrib.utils import save_load_utils

class Agent(threading.Thread):
    def __init__(self, actor, env, id_to_vector_dict, discount_factor, action_size, state_size, random_input_size, scores, episode):
        threading.Thread.__init__(self)

        self.agent_type = ''

        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.image_vectors = []

        self.roll_outs = 5
        self.actor = actor

        self.id_to_vector_dict = id_to_vector_dict
        self.list_of_ids = list(id_to_vector_dict.keys())

        self.discount_factor = discount_factor
        self.action_size = action_size
        self.state_size = state_size
        self.random_input_size = random_input_size

        self.mu = 0
        self.sigma = 2

        self.env = env
        self.env_state = self.env.get_new_state()
        self.env_name = self.env.name

        self.scores = scores
        self.episode = episode

        self.learning_rate = 0.001

    # Thread interactive with environment
    def run(self):
        while True:

            self.env_state.reset()
            state = self.env_state.state

            image_id = random.choice(self.list_of_ids)
            image_vector = self.id_to_vector_dict[image_id]

            score = 0
            while True:
                action, prob = self.get_action(state, image_vector)
                next_state, reward, done = self.env_state.step(action, self, roll_out=True, roll_outs=self.roll_outs, image_vector=image_vector)
                score += reward


                self.memory(state, action, prob, reward, image_vector)

                state = next_state

                if done:
                    self.episode += 1
                    sentence = ' '.join(self.env_state.sentence)
                    print("Episode: " + str(self.episode) + " " + self.name + " produced: " + sentence + " for image: "+str(image_id) + " ->  score : " + str(reward))
                    self.scores.append(reward)
                    self.train_episode()
                    break

            if self.episode % 200 == 0:
                print("++++ Models and scores are saved at episode " + str(self.episode) + " ++++")

                with open("./save_graph/" + self.name + "_scores.bin", "wb") as score_file:
                    pickle.dump(self.scores, score_file)
                self.save_model('./save_model/' + self.name)

    def save_model(self, path):
        save_load_utils.save_all_weights(self.actor, path + "_actor.model")
        save_load_utils.save_all_weights(self.env.discriminator, path + "_discriminator.model")


    def get_action(self, state, image_vector, add_prob = True):
        aprob = self.actor.predict([np.array([state]), np.array([image_vector])], batch_size=1).flatten()
        if add_prob:
            self.probs.append(aprob)
        prob = aprob / np.sum(aprob)
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        return action, prob


    # save <s, a ,r> of each step
    # this is used for calculating discounted rewards
    def memory(self, state, action, prob, reward, image_vector):
        y = np.zeros([self.action_size])
        y[action] = 1
        self.gradients.append(np.array(y).astype('float32') - prob)
        self.states.append(state)
        self.rewards.append(reward)
        self.image_vectors.append(image_vector)

    # update policy network and value network every episode
    def train_episode(self):
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        rewards = rewards - np.mean(rewards)
        std = np.std(rewards)
        if std is not 0:
            rewards = rewards / std
        gradients *= rewards

        # self.X.append([np.squeeze(np.vstack([self.states])),np.array(self.input_vectors)])
        # self.Y.append(self.probs + self.learning_rate * np.squeeze(np.vstack([gradients])))

        x = [np.squeeze(np.vstack([self.states])), np.array(self.image_vectors)]
        y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
        self.actor.fit(x, y, epochs=1, verbose=0)

        #train discriminator
        sentence_image_pairs = self.get_sentence_image_pairs(32)
        self.env.train(sentence_image_pairs)
        self.env.train_with_other_captions(no_of_images=64, runs = 1, training_iterations= 1)

        self.states, self.probs, self.gradients, self.rewards, self.image_vectors = [], [], [], [], []

    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        expected_reward = 0
        for t in range(0, len(rewards)):
            discounted_rewards[t] = rewards[t]-expected_reward*self.discount_factor
            expected_reward = rewards[t]
        return discounted_rewards

    def get_sentence_image_pairs(self, n = 1):
        sentence_image_pairs = []
        for i in range(n):
            # input_vector = np.random.normal(self.mu, self.sigma, self.random_input_size)
            image_id = random.choice(self.list_of_ids)
            image_vector = self.id_to_vector_dict[image_id]
            done = False
            self.env_state.reset()
            state = list(self.env_state.state)
            while not done:
                action, _ = self.get_action(state, image_vector, False)
                next_state, reward, done = self.env_state.step(action, image_vector = image_vector)
                state = next_state
                if done:
                    sentence_image_pairs.append([state, image_vector])
        return np.array(sentence_image_pairs)