#! /usr/bin/env python3

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
import numpy as np
from keras.layers.merge import Concatenate
from sklearn.model_selection import StratifiedKFold
from keras import backend as K

class Other_cnn(object):
    def __init__(self, all_x, all_y, input_shape, num_classes):
        self.all_x = all_x
        self.all_y = all_y
        self.input_shape = input_shape
        self.num_classes = num_classes

    def do_all(self):
        print("INPUT SHAPE: ", self.input_shape)

        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2)
        results = []
        for train, test in kfold.split(self.all_x, self.all_y):
            #filters = (2, 3)
            filters = (5, 7)
            model_input = Input(shape=self.input_shape)
            conv_blocks = []
            for block_size in filters:
                conv = Conv1D(filters=32, kernel_size=block_size, padding='valid', activation='relu', strides=1)(model_input)
                conv = MaxPooling1D()(conv)
                conv = Dropout(0.2)(conv)
                conv1 = Conv1D(filters=64, kernel_size=block_size, padding='valid', activation='relu', strides=1)(model_input)
                conv1 = MaxPooling1D()(conv1)
                conv1 = Dropout(0.2)(conv1)
                conv2 = Conv1D(filters=128, kernel_size=block_size, padding='valid', activation='relu', strides=1)(model_input)
                conv2 = MaxPooling1D()(conv2)
                conv2 = Dropout(0.2)(conv2)
                conv3 = Conv1D(filters=128, kernel_size=block_size, padding='valid', activation='relu', strides=1)(model_input)
                conv3 = Dropout(0.2)(conv3)
                conv3 = Flatten()(conv3)
                conv_blocks.append(conv)
                conv_blocks.append(conv1)
                conv_blocks.append(conv2)
                conv_blocks.append(conv3)

            conc = Concatenate(axis=1)(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
            #conc = Dropout(0.2)(conc)
            conc = Dense(256, activation='relu')(conc)
            conc = Dropout(0.2)(conc)
            model_output = Dense(2, activation='softmax')(conc)

            model = Model(model_input, model_output)
            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

            K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)))
            model.fit(self.all_x[train], keras.utils.to_categorical(self.all_y[train], self.num_classes), batch_size=32, epochs=25, verbose=2, validation_data=(self.all_x[test], keras.utils.to_categorical(self.all_y[test], self.num_classes)))

            score = model.evaluate(self.all_x[test], keras.utils.to_categorical(self.all_y[test], self.num_classes))

            print("loss: ", score[0], "accuracy: ", score[1])
            results.append(score[1])
        print("%.2f%% (+/- %.2f%%)" % (np.mean(results), np.std(results)))
        return results
        
