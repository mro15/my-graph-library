#! /bin/bash

dim="100"
windows="4 12 20"
strategies="no_weight pmi pmi_all"
datasets="polarity webkb r8 20ng"
methods="node2vec"
cuts="5 10 20"

for d in $dim; do
	for w in $windows; do
		for cp in $cuts; do
			for s in $strategies; do
				for dr in $datasets; do
					python3 feature_generator.py --dataset $dr --strategy $s --window $w --emb_dim $d  --cut_percent $cp &
					echo "python3 feature_generator.py --dataset $dr --strategy $s --window $w --emb_dim $d --cut_percent $cp" >> executed_features_next_level.txt &
				done
			done
		done
		wait
	done
done
