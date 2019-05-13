#! /usr/bin/env python3

class Features(object):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def open_file(self, file_type):
        f = open("features/" + self.dataset + "/" + file_type + ".txt", "w")
        return f

    def write_in_file(self, f, features, label):
        line = label + " " + features + "\n" 
        f.write(line)
