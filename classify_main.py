#! /usr/bin/env python3

import argparse
from features.features import Features
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

def read_args():
    parser = argparse.ArgumentParser(description="The parameters are:")
    parser.add_argument('--classifier', type=str, choices=["svm", "mlp", "nb", "rf"], help='classifier name', required=True)   
    parser.add_argument('--train', type=str, help='path for train file', required=True)   
    parser.add_argument('--test', type=str, help='path for test file', required=True)   
    return parser.parse_args()

def read_features(train_path, test_path):
    f = open(train_path, "r")
    x_train = []
    y_train = []
    for line in f:
        splited = line.split()
        y_train.append(int(splited[0]))
        x_train.append([float(x) for x in splited[1:]])
    f.close()

    f = open(test_path, "r")
    x_test = []
    y_test = []
    for line in f:
        splited = line.split()
        y_test.append(int(splited[0]))
        x_test.append([float(x) for x in splited[1:]])
    f.close()

    return x_train, y_train, x_test, y_test


def classify(x_train, y_train, x_test, y_test, classifier):
    c = classifier

    scaler = preprocessing.MaxAbsScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    c.fit(x_train, y_train)
    pred = c.predict(x_test)
    print(c.score(x_test, y_test))
    print(confusion_matrix(y_test, pred))

def main():
    args = read_args()
    x_train, y_train, x_test, y_test = read_features(args.train, args.test)
    
    classifiers = {"nb" : GaussianNB(), "rf": RandomForestClassifier(n_estimators=1000)} 
    print(len(x_train), len(y_train))
    print(len(x_test), len(y_test))
    classify(x_train, y_train, x_test, y_test, classifiers[args.classifier])

if __name__ == "__main__":
    main()
