#! /bin/bash

dim="100"
windows="4 12 20"
strategies="no_weight"
#datasets="r8 webkb polarity"
datasets="20ng"
methods="node2vec"
cuts="0"
time_interval="60"

for d in $dim; do
	for w in $windows; do
		for s in $strategies; do
			for cp in $cuts; do
				for dr in $datasets; do
					mprof run --output time_mem/mprofile.$dr.$s.$w.$cp.dat --interval $time_interval feature_generator.py --dataset $dr --strategy $s --window $w --emb_dim $d  --cut_percent $cp
					echo "python3 feature_generator.py --dataset $dr --strategy $s --window $w --emb_dim $d --cut_percent $cp" >> executed_features_baseline.txt
				done
			done
		done
	done
done
