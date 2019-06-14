#! /usr/bin/env python3

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

class My_cnn(object):
    def __init__(self, train_x, train_y, test_x, test_y, input_shape, num_classes):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.input_shape = input_shape
        self.num_classes = num_classes

    def do_all(self):
        model = Sequential()
        model.add(Conv2D(100, kernel_size=(50,4), activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(50,4)))
        model.add(Conv2D(100, kernel_size=(50, 2),  activation='relu'))
        model.add(MaxPooling2D(pool_size=(50, 2)))
        model.add(Conv2D(100, kernel_size=(50, 4),  activation='relu'))
        model.add(MaxPooling2D(pool_size=(50, 2)))
        model.add(Conv2D(100, kernel_size=(50, 4),  activation='relu'))
        model.add(MaxPooling2D(pool_size=(50, 2)))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

        hist = model.fit(self.train_x, self.train_y, batch_size=100, epochs=50, verbose=1, validation_data=(self.test_x, self.test_y))

        score = model.evaluate(self.test_x, self.test.y)

        print("loss: ", score[0], "accuracy: ", score[1])
        
