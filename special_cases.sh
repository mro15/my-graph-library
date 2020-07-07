#! /bin/bash

dim="100"
windows="4 12 20"
strategies="llr_all"
#datasets="20ng"
datasets="polarity webkb r8"
methods="node2vec"
cuts="30 40 50"
#cuts="10"

for d in $dim; do
	for w in $windows; do
		for cp in $cuts; do
			for s in $strategies; do
				for dr in $datasets; do
					python3 feature_generator.py --dataset $dr --strategy $s --window $w --emb_dim $d  --cut_percent $cp &
					echo "python3 feature_generator.py --dataset $dr --strategy $s --window $w --emb_dim $d --cut_percent $cp" >> executed_features_next_level_special_cases.txt &
				done
				wait
			done
		done
	done
done
