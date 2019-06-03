#! /usr/bin/env python3

import argparse
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import numpy

def read_args():
    parser = argparse.ArgumentParser(description="The parameters are:")
    parser.add_argument('--train', type=str, help='path for train file', required=True)   
    return parser.parse_args()

def read_features(train_path, interval):
    f = open(train_path, "r")
    x_train = []
    y_train = []
    for line in f:
        splited = line.split()
        y_train.append(int(splited[0]))
        x_train.append([float(x) for x in splited[interval]])
    f.close()

    return x_train, y_train

def GridSearch(X_train, y_train):

	C_range = 2. ** numpy.arange(-5,15,2)
	gamma_range = 2. ** numpy.arange(3,-15,-2)
	k = [ 'rbf']
	param_grid = dict(gamma=gamma_range, C=C_range, kernel=k)

	srv = svm.SVC(probability=True)

	grid = GridSearchCV(srv, param_grid, n_jobs=-1, verbose=True)
	grid.fit (X_train, y_train)

	model = grid.best_estimator_

	return grid.best_params_

def write_results(file_name, result):
    with open(file_name, 'w') as f:
        f.write(str(result)+"\n")

def main():
    args = read_args()
    
    x_train, y_train = read_features(args.train, slice(1,-1))
    best = GridSearch(x_train, y_train)
    name = args.train.split("/")[-1] + "all"
    write_results("grid_search_results/"+ name, best)
    
    x_train, y_train = read_features(args.train, slice(1,51))
    best = GridSearch(x_train, y_train)
    name = args.train.split("/")[-1] + "mean"
    write_results("grid_search_results/"+ name, best)
    
    x_train, y_train = read_features(args.train, slice(51,101))
    best = GridSearch(x_train, y_train)
    name = args.train.split("/")[-1] + "median"
    write_results("grid_search_results/"+ name, best)
    
    x_train, y_train = read_features(args.train, slice(101,-1))
    best = GridSearch(x_train, y_train)
    name = args.train.split("/")[-1] + "std"
    write_results("grid_search_results/"+ name, best)
    
    x_train, y_train = read_features(args.train, slice(1,101))
    best = GridSearch(x_train, y_train)
    name = args.train.split("/")[-1] + "mean_median"
    write_results("grid_search_results/"+ name, best)

if __name__ == "__main__":
    main()
