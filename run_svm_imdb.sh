#! /bin/bash

python3 classify_main.py --classifier svm --train features/imdb/no_weight/node2vectrain.txt --test features/imdb/no_weight/node2vectest.txt --features mean --svm_c 2.0 --svm_gamma 8.0

python3 classify_main.py --classifier svm --train features/imdb/no_weight/node2vectrain.txt --test features/imdb/no_weight/node2vectest.txt --features median --svm_c 2.0 --svm_gamma 8.0

python3 classify_main.py --classifier svm --train features/imdb/no_weight/node2vectrain.txt --test features/imdb/no_weight/node2vectest.txt --features std --svm_c 8192.0 --svm_gamma 8.0

python3 classify_main.py --classifier svm --train features/imdb/no_weight/node2vectrain.txt --test features/imdb/no_weight/node2vectest.txt --features mean_median --svm_c 128.0 --svm_gamma 2.0

python3 classify_main.py --classifier svm --train features/imdb/no_weight/node2vectrain.txt --test features/imdb/no_weight/node2vectest.txt --features all --svm_c 512.0 --svm_gamma 2.0
