import numpy as np

class EnvironmentState:

    def __init__(self, env):
        self.env = env
        self.no_of_actions_executed = 0
        self.state = np.zeros(self.env.state_space)
        self.sentence = []
        self.image_vector = None


    def reset(self):
        self.no_of_actions_executed = 0
        self.state = np.zeros(self.env.state_space)
        self.sentence = []
        self.image_vector = None


    def step(self, action, agent = None, roll_out = False, roll_outs = 4, image_vector = None):
        word = self.env.integer_to_word_dict[action]
        embedded_word = self.env.w2v_class.get_embeddings([word])[0]

        if self.image_vector is None:
            self.image_vector = np.reshape(image_vector, (-1,self.env.image_vector_size))

        self.state[self.no_of_actions_executed] = embedded_word
        self.no_of_actions_executed += 1
        self.sentence.append(word)

        return_value = 0
        if self.is_done(word, self.no_of_actions_executed):
            return self.state, self.get_reward(self.state), True

        if roll_out is True and agent is not None:
            return_value = self.roll_out(agent, roll_outs, image_vector)
        return self.state, return_value, False


    def roll_out(self, agent, roll_outs, input_vector):
        rewards = []
        for i in range(roll_outs):
            roll_out_state = list(self.state)
            roll_out_no_of_actions_executed = self.no_of_actions_executed

            while True:
                action, _ = agent.get_action(roll_out_state, input_vector, False)
                roll_out_state, roll_out_no_of_actions_executed, reward, done = self.roll_out_step(action, roll_out_state, roll_out_no_of_actions_executed)
                if done:
                    rewards.append(reward)
                    break
        return np.mean(rewards)


    def roll_out_step(self, action, roll_out_state, roll_out_no_of_actions_executed):
        word = self.env.integer_to_word_dict[action]
        embedded_word = self.env.w2v_class.get_embeddings([word])[0]



        roll_out_state[roll_out_no_of_actions_executed] = embedded_word
        roll_out_no_of_actions_executed += 1
        if self.is_done(word, roll_out_no_of_actions_executed):
            return roll_out_state, roll_out_no_of_actions_executed, self.get_reward(roll_out_state), True

        return roll_out_state, roll_out_no_of_actions_executed, 0, False


    def is_done(self, last_action, actions_executed):
        if actions_executed >= self.env.sentence_length:
            return True
        if last_action == '*e*':
            return True
        return False


    def get_reward(self, state):
        # print(np.array([self.state]))
        prob = self.env.discriminator.predict([np.array([state]), np.array(self.image_vector)])[0][0]
        # print(" ".join(self.sentence)+ " -> discriminator: " +str(prob))
        return prob*50
