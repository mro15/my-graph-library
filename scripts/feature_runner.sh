#! /bin/bash

dim="100"
windows="4 7 20 12"
#strategies="freq freq_all"
strategies="no_weight pmi_1990 pmi_1990_all freq freq_all"
datasets="polarity webkb 20ng"
methods="node2vec"

for d in $dim; do
	for w in $windows; do
		for dr in $datasets; do
			for m in $methods; do
				for s in $strategies; do
					python3 feature_main.py --dataset $dr --method $m --strategy $s --window $w --emb_dim $d &
					echo "python3 feature_main.py --dataset $dr --method $m --strategy $s --window $w --emb_dim $d" >> executed_features.txt &
				done
				wait
			done
		done
	done
done
