#! /usr/bin/env python3

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
import numpy as np
from keras.layers.merge import Concatenate

class My_cnn(object):
    def __init__(self, train_x, train_y, test_x, test_y, input_shape, num_classes):
        self.train_x = train_x
        self.train_y = keras.utils.to_categorical(train_y, num_classes)
        self.test_x = test_x
        self.test_y = keras.utils.to_categorical(test_y, num_classes)
        self.input_shape = input_shape
        self.num_classes = num_classes

    def do_all(self):
        print(self.input_shape, self.train_x.shape)
        filters = (2, 3, 5, 6, 8)
        model_input = Input(shape=self.input_shape)
        conv_blocks = []
        for block_size in filters:
            conv = Conv1D(filters=128, kernel_size=block_size, padding='valid', activation='tanh', strides=1)(model_input)
            conv = Dropout(0.2)(conv)
            conv = GlobalMaxPooling1D()(conv)
            #conv = Flatten()(conv)
            conv_blocks.append(conv)

        conc = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
        conc = Dropout(0.2)(conc)
        conc = Dense(32, activation='relu')(conc)
        model_output = Dense(2, activation='softmax')(conc)

        model = Model(model_input, model_output)
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

        hist = model.fit(self.train_x, self.train_y, batch_size=32, epochs=20, shuffle=True, verbose=2, validation_data=(self.test_x, self.test_y))

        score = model.evaluate(self.test_x, self.test_y)

        print("loss: ", score[0], "accuracy: ", score[1])
        
