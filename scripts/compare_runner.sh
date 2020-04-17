#! /bin/bash

windows="4 7 12 20"
strategies="no_weight pmi_1990 pmi_1990_all freq freq_all"
#strategies="no_weight pmi_2019 pmi_1990 dice llr chi_square"
datasets="webkb polarity 20ng"
#methods="node2vec"

for w in $windows; do
	for dr in $datasets; do
		python3 compare_measures.py --dataset $dr  --strategy pmi_1990 --strategy pmi_1990_all --strategy freq --strategy freq_all --window $w --emb_dim 100
	done
done
