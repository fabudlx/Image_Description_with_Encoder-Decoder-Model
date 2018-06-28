from keras import Input
from keras.engine import Model
from keras.layers import LSTM, Dense, Dropout,merge,TimeDistributed,RepeatVector, multiply, dot, Activation, concatenate, BatchNormalization


def get_discriminator(state_space, image_vector_size):
    print('creating new discriminative model')
    image_input = Input(shape=(image_vector_size,))
    image_dense = Dense(2048, activation='relu', kernel_initializer='glorot_uniform')(image_input)

    image_dropout = Dropout(0.4)(image_dense)

    image_dense2 = Dense(1024, activation='relu', kernel_initializer='glorot_uniform')(image_dropout)

    image_dropout2 = Dropout(0.3)(image_dense2)

    image_dense3 = Dense(512, activation='relu', kernel_initializer='glorot_uniform')(image_dropout2)

    image_normalization = BatchNormalization()(image_dense3)

    state_input = Input(shape=state_space)
    lstm_layer = LSTM(1024, activation='relu', kernel_initializer='glorot_uniform')(state_input)

    sentence_dense = Dense(1024, activation='relu', kernel_initializer='glorot_uniform')(lstm_layer)

    sentence_dropout = Dropout(0.4)(sentence_dense)

    sentence_dense2 = Dense(1024, activation='relu', kernel_initializer='glorot_uniform')(sentence_dropout)

    sentence_dropout2 = Dropout(0.3)(sentence_dense2)

    sentence_dense3 = Dense(512, activation='relu', kernel_initializer='glorot_uniform')(sentence_dropout2)

    sentence_normalization = BatchNormalization()(sentence_dense3)

    dot_product = dot([sentence_normalization, image_normalization], axes=1, normalize=False)
    fake_or_real_layer = Activation('sigmoid')(dot_product)

    model = Model(inputs=[state_input, image_input], outputs=fake_or_real_layer)

    model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model



# approximate policy and value using Neural Network
# actor -> state is input and probability of each action is output of network
def build_actor_model(state_space, action_size, image_vector_size):

    image_input = Input(shape=(image_vector_size,))

    dense_image_vector = Dense(256, activation='relu', kernel_initializer='glorot_uniform')(image_input)

    repeated_image_vector = RepeatVector(state_space[0])(dense_image_vector)

    sentence_input = Input(shape=state_space)

    sentence_lstm = LSTM(512,return_sequences=True)(sentence_input)

    time_distributed_lstm = TimeDistributed(Dense(256))(sentence_lstm)

    merger = concatenate([repeated_image_vector, time_distributed_lstm])

    lstm = LSTM(1028, return_sequences=True)(merger)

    lstm = LSTM(1028, return_sequences=True)(lstm)


    activation = Dense(action_size, activation='softmax')(lstm)

    model = Model([image_input, sentence_input], activation)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model