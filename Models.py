from keras import Input
from keras.engine import Model
from keras.layers import LSTM, Dense, Dropout,merge,TimeDistributed,RepeatVector, multiply, dot, Activation, concatenate, BatchNormalization


# approximate policy and value using Neural Network
# actor -> state is input and probability of each action is output of network
def build_actor_model(state_space, action_size, image_vector_size):

    image_input = Input(shape=(image_vector_size,))

    dense_image_vector = Dense(512, activation='relu', kernel_initializer='glorot_uniform')(image_input)

    repeated_image_vector = RepeatVector(state_space[0])(dense_image_vector)

    sentence_input = Input(shape=state_space)

    sentence_lstm = LSTM(1024,return_sequences=True)(sentence_input)

    time_distributed_lstm = TimeDistributed(Dense(512))(sentence_lstm)

    merger = concatenate([repeated_image_vector, time_distributed_lstm])

    lstm = LSTM(1024, return_sequences=True)(merger)

    dropout = Dropout(0.3)(lstm)

    lstm1 = LSTM(1024, return_sequences=True)(dropout)

    dropout2 = Dropout(0.3)(lstm1)

    dense = Dense(2048, activation='relu', kernel_initializer='glorot_uniform')(dropout2)

    activation = Dense(action_size, activation='softmax')(dense)

    model = Model([image_input, sentence_input], activation)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model