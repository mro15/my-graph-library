#! /bin/bash

dim="100 300"
windows="4 7 20 12"
strategies="no_weight pmi_2019 pmi_1990 dice llr chi_square"

for d in $dim; do
	for w in $windows; do
		for s in $strategies; do
			python3 feature_main.py --dataset polarity --method node2vec --strategy $s --window $w --emb_dim $d
			echo "python3 feature_main.py --dataset polarity --method node2vec --strategy $s --window $w --emb_dim $d" >> executed_features.txt
		done
	done
done
