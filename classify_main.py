#! /usr/bin/env python3

import argparse
from features.features import Features
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

def read_args():
    parser = argparse.ArgumentParser(description="The parameters are:")
    parser.add_argument('--classifier', type=str, choices=["svm", "mlp", "nb", "rf"], help='classifier name', required=True)   
    parser.add_argument('--features', type=str, choices=["all", "mean", "median", "std", "mean_median"], help='features', required=True)   
    parser.add_argument('--train', type=str, help='path for train file', required=True)   
    parser.add_argument('--svm_c', type=float, help='C param for SVM', required=False)   
    parser.add_argument('--svm_gamma', type=float, help='gamma param for SVM', required=False)   
    parser.add_argument('--test', type=str, help='path for test file', required=True)   
    return parser.parse_args()

def read_features(train_path, test_path, interval):
    f = open(train_path, "r")
    x_train = []
    y_train = []
    for line in f:
        splited = line.split()
        y_train.append(int(splited[0]))
        x_train.append([float(x) for x in splited[interval]])
    f.close()

    f = open(test_path, "r")
    x_test = []
    y_test = []
    for line in f:
        splited = line.split()
        y_test.append(int(splited[0]))
        x_test.append([float(x) for x in splited[interval]])
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

    if (args.classifier == "svm" and (not args.svm_c or not args.svm_gamma)):
        print("For svm classifier svm_c and svm_gamma are required \n")
        exit()

    features = {"all" : slice(1,-1), "mean" : slice(1,51), "median" : slice(51,101), "std" : slice(101,-1), "mean_median" : slice(1,101)}

    x_train, y_train, x_test, y_test = read_features(args.train, args.test, features[args.features])
    
    classifiers = {"nb" : GaussianNB(), "rf": RandomForestClassifier(n_estimators=1000), "svm" : svm.SVC(kernel='linear', C=args.svm_c, gamma=args.svm_gamma, probability=True)} 
    classify(x_train, y_train, x_test, y_test, classifiers[args.classifier])

if __name__ == "__main__":
    main()