#! /usr/bin/env python3

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
import numpy as np
from keras.layers.merge import Concatenate
from sklearn.model_selection import StratifiedKFold
from keras import backend as K
from sklearn.metrics import f1_score

class My_cnn(object):
    def __init__(self, all_x, all_y, input_shape, num_classes, pooling_type):
        self.all_x = all_x
        self.all_y = all_y
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.pooling_type = pooling_type

    def get_vectors_x(self, t):
        vec_train = []
        for i in self.all_x[t]:
            vec_train.append(i.A)
        return np.array(vec_train)

    def do_all(self):
        print("INPUT SHAPE: ", self.input_shape)
        fold = 1
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2)
        results = []
        f1_results = []
        for train, test in kfold.split(self.all_x, self.all_y):
            print("RUNNING FOLD ", fold)
            filters = (2, 3)
            model_input = Input(shape=self.input_shape)
            conv_blocks = []
            for block_size in filters:
                conv = Conv1D(filters=32, kernel_size=block_size, padding='valid', activation='relu', strides=1)(model_input)
                conv = Dropout(0.2)(conv)
                if self.pooling_type == "max":
                    conv = MaxPooling1D()(conv)
                    conv = Flatten()(conv)
                else:
                    conv = GlobalMaxPooling1D()(conv)
                conv1 = Conv1D(filters=64, kernel_size=block_size, padding='valid', activation='relu', strides=1)(model_input)
                conv1 = Dropout(0.2)(conv1)
                if self.pooling_type == "max":
                    conv1 = MaxPooling1D()(conv1)
                    conv1 = Flatten()(conv1)
                else:
                    conv1 = GlobalMaxPooling1D()(conv1)
                conv2 = Conv1D(filters=128, kernel_size=block_size, padding='valid', activation='relu', strides=1)(model_input)
                conv2 = Dropout(0.2)(conv2)
                if self.pooling_type == "max":
                    conv2 = MaxPooling1D()(conv2)
                    conv2 = Flatten()(conv2)
                else:
                    conv2 = GlobalMaxPooling1D()(conv2)
                conv_blocks.append(conv2)
                conv_blocks.append(conv1)
                conv_blocks.append(conv)

            conc = Concatenate(axis=1)(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
            #conc = Dropout(0.2)(conc)
            conc = Dense(256, activation='relu')(conc)
            conc = Dropout(0.2)(conc)
            model_output = Dense(self.num_classes, activation='softmax')(conc)

            model = Model(model_input, model_output)
            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

            data_train = self.get_vectors_x(train)
            data_test = self.get_vectors_x(test)
            
            K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=12, inter_op_parallelism_threads=12)))
            model.fit(data_train, keras.utils.to_categorical(self.all_y[train], self.num_classes), batch_size=32, epochs=20, verbose=2, validation_data=(data_test, keras.utils.to_categorical(self.all_y[test], self.num_classes)))

            score = model.evaluate(data_test, keras.utils.to_categorical(self.all_y[test], self.num_classes))

            results.append(score[1])
            y_prob = model.predict(data_test)
            y_pred = np.argmax(y_prob, axis=1)
            f1 = f1_score(self.all_y[test], y_pred, average='macro')
            f1_results.append(f1)
            all_x = None
            all_y = None
            K.clear_session()
            print("loss: ", score[0], "accuracy: ", score[1], "f1: ", f1)
            fold += 1
        print("MEAN: ", np.mean(results), "STD: ", np.std(results), "MEAN F1: ", np.mean(f1_results), "STD F1: ", np.std(f1_results))
        return results, f1_results
        
